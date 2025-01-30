#![allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use image::GenericImageView;
use std::{
    borrow::Cow,
    fs,
    io::Cursor,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use wgpu::{LoadOp, PipelineCompilationOptions, Sampler, ShaderSource, StoreOp};
use winit::{
    application::ApplicationHandler,
    error::EventLoopError,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
use zip::ZipArchive;

const IMAGE_ZIP_URL: &str = "http://m.adept.care/display_tv/MH.zip";
const IMAGE_ZIP_PATH: &str = "images.zip";
const IMAGE_DIR_PATH: &str = "images";

const TIME_BETWEEN_IMAGES: f64 = 10.0;
const TRANSITION_TIME: f64 = 1.0;

const WAIT_TIME: Duration = Duration::from_millis(66); // Frame time between refreshes

fn main() -> Result<(), EventLoopError> {
    // Grab new images on startup
    ensure_latest_images().unwrap();

    // Start the event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut App {
        window: None,
        wgpu_ctx: None,
        start_time: Instant::now(),
    })
}

struct App<'window> {
    window: Option<Arc<Window>>,
    wgpu_ctx: Option<WgpuCtx<'window>>,
    start_time: Instant,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Slideshow")
                            .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))),
                    )
                    .unwrap(),
            );
            self.window = Some(Arc::clone(&window));
            self.wgpu_ctx = Some(WgpuCtx::new(&window));
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let start_time = Instant::now();
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let cycle_position = elapsed % TIME_BETWEEN_IMAGES;
        let in_transition = cycle_position >= (TIME_BETWEEN_IMAGES - TRANSITION_TIME);

        // Calculate next wake time
        let next_wake = if in_transition {
            // Wake more frequently during transition (every WAIT_TIME)
            Instant::now() + WAIT_TIME
        } else {
            // Wake at transition start with small buffer
            let next_transition =
                elapsed - cycle_position + (TIME_BETWEEN_IMAGES - TRANSITION_TIME);
            self.start_time + Duration::from_secs_f64(next_transition) + WAIT_TIME * 2
        };

        // Update image if needed
        let update_start = Instant::now();
        let wgpu_ctx = self.wgpu_ctx.as_mut().unwrap();
        let next_image = ((elapsed / TIME_BETWEEN_IMAGES) as usize) % wgpu_ctx.image_paths.len();
        if next_image != wgpu_ctx.current_image_index {
            wgpu_ctx.update_textures(next_image);
        }
        let update_duration = update_start.elapsed();

        event_loop.set_control_flow(ControlFlow::WaitUntil(next_wake));

        let draw_start = Instant::now();
        if let Some(wgpu_ctx) = self.wgpu_ctx.as_mut() {
            wgpu_ctx.draw(elapsed);
        }
        let draw_duration = draw_start.elapsed();

        let total_duration = start_time.elapsed();
        println!(
            "about_to_wait: {:.2}ms (update: {:.2}ms, draw: {:.2}ms)",
            total_duration.as_secs_f64() * 1000.0,
            update_duration.as_secs_f64() * 1000.0,
            draw_duration.as_secs_f64() * 1000.0
        );
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let (Some(wgpu_ctx), Some(window)) =
                    (self.wgpu_ctx.as_mut(), self.window.as_ref())
                {
                    wgpu_ctx.surface_config.width = new_size.width.max(1);
                    wgpu_ctx.surface_config.height = new_size.height.max(1);
                    wgpu_ctx
                        .surface
                        .configure(&wgpu_ctx.device, &wgpu_ctx.surface_config);
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(wgpu_ctx) = self.wgpu_ctx.as_mut() {
                    wgpu_ctx.draw(self.start_time.elapsed().as_secs_f64());
                }
            }
            _ => (),
        }
    }
}

struct WgpuCtx<'window> {
    surface: wgpu::Surface<'window>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    front_texture: TextureResources,
    back_texture: TextureResources,
    back_texture_size: (u32, u32),
    texture_sampler: Sampler,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: wgpu::BindGroup,
    image_paths: Vec<String>,
    current_image_index: usize,
}

impl WgpuCtx<'_> {
    fn new(window: &Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(window)).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        let mut limits =
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
        limits.max_push_constant_size = 16;
        let mut features = wgpu::Features::default();
        features.set(wgpu::Features::PUSH_CONSTANTS, true);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                label: None,
            },
            None,
        ))
        .unwrap();

        let size = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .unwrap();
        surface_config.present_mode = wgpu::PresentMode::Mailbox;
        surface.configure(&device, &surface_config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            label: None,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Binding for the texture sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Binding for the first texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Binding for the second texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
                label: None,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&texture_bind_group_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::FRAGMENT,
                        range: 0..16,
                    }],
                    label: None,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            multisample: wgpu::MultisampleState::default(),
            depth_stencil: None,
            multiview: None,
            cache: None,
            label: None,
        });

        let image_paths = std::fs::read_dir(IMAGE_DIR_PATH)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|e| e.path().to_str().unwrap().to_owned()))
            .collect::<Vec<_>>();

        // Ensure there are available images
        assert!(
            !image_paths.is_empty(),
            "No images found in 'images' directory"
        );

        let current_image = image::open(&image_paths[0]).unwrap();
        let next_image = image::open(&image_paths[1]).unwrap();

        let front_texture = create_texture_resources(&device, &queue, &current_image);
        let back_texture = create_texture_resources(&device, &queue, &next_image);
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let texture_bind_group = update_bind_groups(
            &device,
            &texture_bind_group_layout,
            &front_texture,
            &back_texture,
            &texture_sampler,
        );

        WgpuCtx {
            surface,
            surface_config,
            device,
            queue,
            render_pipeline,
            front_texture,
            back_texture,
            back_texture_size: current_image.dimensions(),
            texture_sampler,
            texture_bind_group_layout,
            texture_bind_group,
            image_paths,
            current_image_index: 0,
        }
    }

    fn update_textures(&mut self, next_image_index: usize) {
        std::mem::swap(&mut self.front_texture, &mut self.back_texture);
        self.current_image_index = next_image_index;
        let img_path = &self.image_paths[next_image_index];
        let next_image = image::open(img_path).unwrap();
        self.back_texture = create_texture_resources(&self.device, &self.queue, &next_image);

        let (width, height) = next_image.dimensions();

        if (width, height) == self.back_texture_size {
            // Reuse existing texture
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.back_texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &next_image.to_rgba8(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            // Create new texture
            self.back_texture = create_texture_resources(&self.device, &self.queue, &next_image);
            self.back_texture_size = (width, height);
        }

        self.texture_bind_group = update_bind_groups(
            &self.device,
            &self.texture_bind_group_layout,
            &self.front_texture,
            &self.back_texture,
            &self.texture_sampler,
        );
    }

    fn draw(&self, elapsed: f64) {
        let total_start = Instant::now();

        // Acquire surface texture
        let texture_start = Instant::now();
        let surface_texture = self.surface.get_current_texture().unwrap();
        let texture_time = texture_start.elapsed();

        // Create command encoder
        let encoder_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let encoder_time = encoder_start.elapsed();

        // Begin the render pass
        let render_pass_start = Instant::now();
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: LoadOp::default(),
                        store: StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                label: None,
            });
            // Set up pipeline and bind groups
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.texture_bind_group, &[]);

            let transition = ((elapsed % TIME_BETWEEN_IMAGES)
                - (TIME_BETWEEN_IMAGES - TRANSITION_TIME))
                / TRANSITION_TIME;
            let transition = transition.clamp(0.0, 1.0) as f32;
            rpass.set_push_constants(
                wgpu::ShaderStages::FRAGMENT,
                0,
                bytemuck::cast_slice(&[transition]),
            );

            rpass.draw(0..4, 0..1);
        }
        let render_pass_time = render_pass_start.elapsed();

        // Submit commands
        let submit_start = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        let submit_time = submit_start.elapsed();

        // Present frame
        let present_start = Instant::now();
        surface_texture.present();
        let present_time = present_start.elapsed();

        let total_time = total_start.elapsed();
        println!(
            "Draw: {:.2}ms (tex: {:.2}, enc: {:.2}, pass: {:.2}, sub: {:.2}, pres: {:.2})",
            total_time.as_secs_f64() * 1000.0,
            texture_time.as_secs_f64() * 1000.0,
            encoder_time.as_secs_f64() * 1000.0,
            render_pass_time.as_secs_f64() * 1000.0,
            submit_time.as_secs_f64() * 1000.0,
            present_time.as_secs_f64() * 1000.0
        );
    }
}

fn update_bind_groups(
    device: &wgpu::Device,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    front_texture: &TextureResources,
    back_texture: &TextureResources,
    texture_sampler: &Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(texture_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&front_texture.texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&back_texture.texture_view),
            },
        ],
        label: None,
    })
}

struct TextureResources {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
}

fn create_texture_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &image::DynamicImage,
) -> TextureResources {
    let (width, height) = image.dimensions();
    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        view_formats: &[],
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: None,
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &image.to_rgba8(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        texture_size,
    );

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    TextureResources {
        texture,
        texture_view,
    }
}

fn ensure_latest_images() -> Result<(), Box<dyn std::error::Error>> {
    let zip_path = Path::new(IMAGE_ZIP_PATH);
    if zip_path.exists() {
        let modified = fs::metadata(zip_path)?.modified()?;
        if modified.elapsed()? <= Duration::from_secs(3600) {
            return Ok(());
        }
    }

    // Download ZIP
    let client = reqwest::blocking::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;
    let bytes = client.get(IMAGE_ZIP_URL).send()?.bytes()?;

    // Write the ZIP archive to disk
    fs::write(zip_path, &bytes)?;

    // Clear the images directory
    if Path::new(IMAGE_DIR_PATH).exists() {
        fs::remove_dir_all(IMAGE_DIR_PATH)?;
    }
    fs::create_dir_all(IMAGE_DIR_PATH)?;

    let mut zip = ZipArchive::new(Cursor::new(bytes))?;
    for i in 0..zip.len() {
        let mut file = zip.by_index(i)?;
        let mut outfile = fs::File::create(Path::new(IMAGE_DIR_PATH).join(file.name()))?;
        std::io::copy(&mut file, &mut outfile)?;
    }

    Ok(())
}

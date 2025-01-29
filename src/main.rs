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
use wgpu::{PipelineCompilationOptions, ShaderSource};
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

const TIME_BETWEEN_IMAGES: f64 = 4.0;
const TRANSITION_TIME: f64 = 1.0;

fn main() -> Result<(), EventLoopError> {
    // Grab new images on startup
    ensure_latest_images().unwrap();

    // Start the event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        wgpu_ctx: None,
        start_time: Instant::now(),
    };
    event_loop.run_app(&mut app)
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

                    // Ensure continuous request for redraw
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
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
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: wgpu::BindGroup,
    mix_factor_buffer: wgpu::Buffer,
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
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::default(),
                required_limits:
                    wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::default(),
                label: None,
            },
            None,
        ))
        .unwrap();

        let size = window.inner_size();
        let surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .unwrap();
        surface.configure(&device, &surface_config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            label: None,
        });

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Binding for the first texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding for the first sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                // Binding for the second sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding for the mix factor uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<f32>() as _),
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
                    push_constant_ranges: &[],
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

        let mix_factor_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
            label: None,
        });
        let texture_bind_group = update_bind_groups(
            &device,
            &texture_bind_group_layout,
            &front_texture,
            &back_texture,
            &mix_factor_buffer,
        );

        WgpuCtx {
            surface,
            surface_config,
            device,
            queue,
            render_pipeline,
            front_texture,
            back_texture,
            texture_bind_group_layout,
            texture_bind_group,
            mix_factor_buffer,
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

        self.texture_bind_group = update_bind_groups(
            &self.device,
            &self.texture_bind_group_layout,
            &self.front_texture,
            &self.back_texture,
            &self.mix_factor_buffer,
        );
    }

    fn draw(&mut self, elapsed_time: f64) {
        // Track ticker
        let next_image = ((elapsed_time / TIME_BETWEEN_IMAGES) as usize) % self.image_paths.len();
        let needs_update = next_image != self.current_image_index;
        if needs_update {
            self.current_image_index = next_image;
            self.update_textures(next_image);
        }
        let transition = (elapsed_time % TIME_BETWEEN_IMAGES) / TRANSITION_TIME;
        if transition > 1.5 && !needs_update {
            return;
        }

        self.queue.write_buffer(
            &self.mix_factor_buffer,
            0,
            bytemuck::cast_slice(&[transition.min(1.0) as f32]),
        );

        // Acquire the current texture for rendering
        let surface_texture = self.surface.get_current_texture().unwrap();
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Start the command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // Begin the render pass
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                label: None,
            });
            // Set up pipeline and bind groups
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.texture_bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }

        // Submit the commands for rendering
        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }
}

fn update_bind_groups(
    device: &wgpu::Device,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    front_texture: &TextureResources,
    back_texture: &TextureResources,
    mix_factor_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&front_texture.texture_view), // Front texture
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&front_texture.sampler), // Front sampler
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&back_texture.texture_view), // Back texture
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&back_texture.sampler), // Back sampler
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: mix_factor_buffer,
                    offset: 0,
                    size: None,
                }), // Mix Factor
            },
        ],
        label: None,
    })
}

struct TextureResources {
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

fn create_texture_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &image::DynamicImage,
) -> TextureResources {
    let rgba = image.to_rgba8();
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
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        texture_size,
    );

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

    TextureResources {
        texture_view,
        sampler,
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

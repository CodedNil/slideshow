#![allow(
    clippy::missing_panics_doc,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use gl::types::{GLfloat, GLint, GLuint};
use glutin::{
    config::ConfigTemplateBuilder,
    context::{ContextApi, ContextAttributesBuilder, PossiblyCurrentContext},
    display::GetGlDisplay,
    prelude::{GlDisplay, NotCurrentGlContext},
    surface::{GlSurface, Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface},
};
use glutin_winit::{DisplayBuilder, GlWindow};
use image::{DynamicImage, RgbaImage};
use pdfium_render::prelude::{PdfRenderConfig, Pdfium};
use raw_window_handle::HasWindowHandle;
use std::{
    ffi::{CStr, CString},
    fs::{self, File},
    io::{self, BufReader},
    num::NonZeroU32,
    path::Path,
    time::{Duration, Instant},
};
use ureq::{tls::TlsConfig, Agent};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

const IMAGE_PDF_URL: &str = "http://m.adept.care/display_tv/MH.pdf";
const IMAGE_PDF_PATH: &str = "images.pdf";
const IMAGE_DIR_PATH: &str = "images";

const TIME_BETWEEN_IMAGES: f64 = 10.0;
const TRANSITION_TIME: f64 = 1.0;

const WAIT_TIME: Duration = Duration::from_millis(50); // Frame time between refreshes

pub mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
    pub use Gles2 as Gl;
}

pub fn main() {
    // Grab new images on startup
    ensure_latest_images().unwrap();
    let mut image_paths = fs::read_dir(IMAGE_DIR_PATH)
        .unwrap()
        .filter_map(|e| {
            e.ok()
                .and_then(|e| e.path().to_str().map(std::borrow::ToOwned::to_owned))
        })
        .collect::<Vec<_>>();
    image_paths.sort_by_key(|s| {
        let first = s.split('/').last().unwrap();
        first.split('.').next().unwrap().parse::<usize>().unwrap()
    });

    // Start the event loop
    let event_loop = EventLoop::new().unwrap();
    if let Err(e) = event_loop.run_app(&mut App {
        gl_display: None,
        gl_context: None,
        state: None,
        renderer: None,
        image_paths,
        current_image_index: 0,
        start_time: Instant::now(),
    }) {
        eprintln!("Error: {e}");
    }
}

struct App {
    renderer: Option<Renderer>,
    state: Option<AppState>,
    gl_context: Option<PossiblyCurrentContext>,
    gl_display: Option<glutin_winit::DisplayBuilder>,
    image_paths: Vec<String>,
    current_image_index: usize,
    start_time: Instant,
}

struct AppState {
    gl_surface: Surface<WindowSurface>,
    window: Window,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let display_builder = self.gl_display.take().unwrap_or_else(|| {
            DisplayBuilder::new().with_window_attributes(Some(
                Window::default_attributes()
                    .with_title("Slideshow")
                    .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))),
            ))
        });

        // Build window and GL config
        let (window, gl_config) = display_builder
            .build(event_loop, ConfigTemplateBuilder::new(), |mut configs| {
                configs.next().unwrap()
            })
            .expect("Failed to create window and GL config");
        let window = window.unwrap();

        // Create GL surface
        let attrs = window
            .build_surface_attributes(SurfaceAttributesBuilder::default())
            .expect("Failed to build surface attributes");
        let gl_surface = unsafe {
            gl_config
                .display()
                .create_window_surface(&gl_config, &attrs)
                .expect("Failed to create GL surface")
        };

        // Create context with GLES
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(None))
            .build(Some(window.window_handle().unwrap().as_raw()));
        // Store the context once it's current
        self.gl_context = Some(
            unsafe {
                gl_config
                    .display()
                    .create_context(&gl_config, &context_attributes)
                    .unwrap()
            }
            .make_current(&gl_surface)
            .unwrap(),
        );

        self.renderer.get_or_insert_with(|| {
            Renderer::new(
                &gl_config.display(),
                &self.image_paths,
                self.current_image_index,
            )
        });

        // Attempt to set VSync using the context and surface
        gl_surface
            .set_swap_interval(
                self.gl_context.as_ref().unwrap(),
                SwapInterval::Wait(NonZeroU32::new(1).unwrap()),
            )
            .unwrap();

        self.state = Some(AppState { gl_surface, window });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) if size.width != 0 && size.height != 0 => {
                let state = self.state.as_ref().unwrap();
                state.gl_surface.resize(
                    self.gl_context.as_ref().unwrap(),
                    size.width.try_into().unwrap(),
                    size.height.try_into().unwrap(),
                );
                unsafe {
                    self.renderer.as_ref().unwrap().gl.Viewport(
                        0,
                        0,
                        size.width as i32,
                        size.height as i32,
                    );
                };
                state.window.request_redraw();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => (),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let cycle_position = elapsed % TIME_BETWEEN_IMAGES;
        let in_transition = cycle_position >= (TIME_BETWEEN_IMAGES - TRANSITION_TIME);

        // Update image if needed
        let next_image = ((elapsed / TIME_BETWEEN_IMAGES) as usize) % self.image_paths.len();
        if self.current_image_index != next_image {
            self.current_image_index = next_image;
            if let Some(renderer) = &mut self.renderer {
                renderer.update_textures(&self.image_paths, self.current_image_index);
            }
        }

        // Draw image
        if let Some(state) = &self.state {
            if let Some(renderer) = &mut self.renderer {
                renderer.draw(
                    (((elapsed % TIME_BETWEEN_IMAGES) - (TIME_BETWEEN_IMAGES - TRANSITION_TIME))
                        / TRANSITION_TIME)
                        .clamp(0.0, 1.0) as f32,
                );
            }
            state.window.request_redraw();
            state
                .gl_surface
                .swap_buffers(self.gl_context.as_ref().unwrap())
                .unwrap();
        }

        // Calculate next wake time
        event_loop.set_control_flow(ControlFlow::WaitUntil(if in_transition {
            // Wake more frequently during transition (every WAIT_TIME)
            Instant::now() + WAIT_TIME
        } else {
            // Wake at transition start with small buffer
            let next_transition =
                elapsed - cycle_position + (TIME_BETWEEN_IMAGES - TRANSITION_TIME);
            self.start_time + Duration::from_secs_f64(next_transition) + WAIT_TIME * 2
        }));
    }
}

pub struct Renderer {
    program: GLuint,
    vao: gl::types::GLuint,
    textures: [GLuint; 2],
    gl: gl::Gl,
    transition_loc: GLint,
}

impl Renderer {
    pub fn new<D: GlDisplay>(gl_display: &D, image_paths: &[String], start_index: usize) -> Self {
        let gl = gl::Gl::load_with(|s| {
            gl_display
                .get_proc_address(CString::new(s).unwrap().as_c_str())
                .cast()
        });
        for (param, prefix) in [
            (gl::RENDERER, "Running on"),
            (gl::VERSION, "OpenGL Version"),
            (gl::SHADING_LANGUAGE_VERSION, "Shaders version"),
        ] {
            if let Some(s) = {
                unsafe {
                    let s = gl.GetString(param);
                    (!s.is_null()).then(|| CStr::from_ptr(s.cast()))
                }
            } {
                println!("{} {}", prefix, s.to_string_lossy());
            }
        }

        unsafe {
            // Create the shader program
            let program = gl.CreateProgram();
            for (shader_type, source) in [
                (gl::VERTEX_SHADER, VERTEX_SHADER),
                (gl::FRAGMENT_SHADER, FRAGMENT_SHADER),
            ] {
                let shader = gl.CreateShader(shader_type);
                gl.ShaderSource(
                    shader,
                    1,
                    [source.as_ptr().cast()].as_ptr(),
                    std::ptr::null(),
                );
                gl.CompileShader(shader);
                gl.AttachShader(program, shader);
                gl.DeleteShader(shader);
            }
            gl.LinkProgram(program);
            gl.UseProgram(program);

            // Retrieve and set the uniform locations for the texture samplers
            gl.Uniform1i(
                gl.GetUniformLocation(program, c"u_texture0".as_ptr().cast::<_>()),
                0,
            );
            gl.Uniform1i(
                gl.GetUniformLocation(program, c"u_texture1".as_ptr().cast::<_>()),
                1,
            );

            // Create the vertex array
            let mut vertex_array_object = 0;
            gl.GenVertexArrays(1, &mut vertex_array_object);
            gl.BindVertexArray(vertex_array_object);
            let mut vertex_buffer_object = 0;
            gl.GenBuffers(1, &mut vertex_buffer_object);
            gl.BindBuffer(gl::ARRAY_BUFFER, vertex_buffer_object);
            gl.BufferData(
                gl::ARRAY_BUFFER,
                (VERTEX_DATA.len() * std::mem::size_of::<f32>()) as isize,
                VERTEX_DATA.as_ptr().cast(),
                gl::STATIC_DRAW,
            );
            for (name, offset) in [
                (c"position", std::ptr::null()),
                (c"tex_coord", (8 as *const ()).cast()),
            ] {
                let loc = gl.GetAttribLocation(program, name.as_ptr().cast());
                gl.VertexAttribPointer(loc as u32, 2, gl::FLOAT, 0, 16, offset);
                gl.EnableVertexAttribArray(loc as u32);
            }

            // Retrieve the location of the transition uniform
            let transition_loc = gl.GetUniformLocation(program, c"transition".as_ptr().cast::<_>());

            // Load the textures
            let mut textures = [0, 0];
            gl.GenTextures(2, textures.as_mut_ptr());
            for (index, image_index) in [
                (textures[0], start_index),
                (textures[1], (start_index + 1) % image_paths.len()),
            ] {
                gl.BindTexture(gl::TEXTURE_2D, index);
                load_texture_from_path(&gl, &image_paths[image_index]);
            }

            Self {
                program,
                vao: vertex_array_object,
                textures,
                gl,
                transition_loc,
            }
        }
    }

    pub fn update_textures(&mut self, image_paths: &[String], current_index: usize) {
        self.textures.swap(0, 1);
        unsafe {
            self.gl.DeleteTextures(1, &self.textures[1]);
            self.gl.GenTextures(1, &mut self.textures[1]);
            self.gl.BindTexture(gl::TEXTURE_2D, self.textures[1]);
            load_texture_from_path(
                &self.gl,
                &image_paths[(current_index + 1) % image_paths.len()],
            );
        }
    }

    pub fn draw(&self, transition_value: GLfloat) {
        unsafe {
            self.gl.UseProgram(self.program);
            self.gl.BindVertexArray(self.vao);

            for (i, &texture) in self.textures.iter().enumerate() {
                self.gl.ActiveTexture(gl::TEXTURE0 + i as u32);
                self.gl.BindTexture(gl::TEXTURE_2D, texture);
            }

            self.gl.Uniform1f(self.transition_loc, transition_value);

            self.gl.ClearColor(0.0, 0.0, 0.0, 1.0);
            self.gl.Clear(gl::COLOR_BUFFER_BIT);
            self.gl.DrawArrays(gl::TRIANGLE_STRIP, 0, 4);
        }
    }
}

fn load_texture_from_path(gl: &gl::Gl, path: &str) {
    let img = image::open(path).unwrap().flipv().to_rgba8();
    let (w, h) = img.dimensions();
    unsafe {
        gl.TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RGBA as i32,
            w as i32,
            h as i32,
            0,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            img.as_raw().as_ptr().cast(),
        );
        gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
    }
}

#[rustfmt::skip]
static VERTEX_DATA: [f32; 16] = [
    -1.0, -1.0, 0.0, 0.0, // Bottom-left
     1.0, -1.0, 1.0, 0.0, // Bottom-right
    -1.0,  1.0, 0.0, 1.0, // Top-left
     1.0,  1.0, 1.0, 1.0, // Top-right
];

const VERTEX_SHADER: &[u8] = b"
#version 100
attribute vec2 position;
attribute vec2 tex_coord;
varying vec2 v_tex;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_tex = tex_coord;
}\0";

const FRAGMENT_SHADER: &[u8] = b"
#version 100
precision mediump float;
varying vec2 v_tex;
uniform sampler2D u_texture0, u_texture1;
uniform float transition;

void main() {
    vec4 color0 = texture2D(u_texture0, v_tex);
    vec4 color1 = texture2D(u_texture1, v_tex);
    gl_FragColor = mix(color0, color1, transition);
}\0";

/// Ensures the latest images are available from the server and unpacks them to a directory.
///
/// If the images have been updated in the last hour, this does nothing.
/// Otherwise, it downloads the latest images from `IMAGE_PDF_URL`, unpacks them to `IMAGE_DIR_PATH`
fn ensure_latest_images() -> Result<()> {
    let pdf_path = Path::new(IMAGE_PDF_PATH);
    if pdf_path.exists() {
        let modified = fs::metadata(pdf_path)?.modified()?;
        if modified.elapsed()? <= Duration::from_secs(3600) {
            return Ok(());
        }
    }

    // Download PDF using ureq
    let tls_config = TlsConfig::builder().disable_verification(true).build();
    let agent = Agent::from(Agent::config_builder().tls_config(tls_config).build());
    let mut res = agent.get(IMAGE_PDF_URL).call()?;
    let mut reader = BufReader::new(
        res.body_mut()
            .with_config()
            .limit(30 * 1024 * 1024)
            .reader(),
    );

    // Write the PDF to disk
    let mut file = File::create(pdf_path)?;
    io::copy(&mut reader, &mut file)?;

    // Clear the images directory
    if Path::new(IMAGE_DIR_PATH).exists() {
        fs::remove_dir_all(IMAGE_DIR_PATH)?;
    }
    fs::create_dir_all(IMAGE_DIR_PATH)?;

    // Load the pdf and export each page as an image
    let pdfium = Pdfium::default();
    let document = pdfium.load_pdf_from_file(pdf_path, None)?;
    for (index, page) in document.pages().iter().enumerate() {
        let page = page.render_with_config(&PdfRenderConfig::new().set_target_width(1920))?;
        RgbaImage::from_raw(
            page.width() as u32,
            page.height() as u32,
            page.as_rgba_bytes(),
        )
        .map(DynamicImage::ImageRgba8)
        .unwrap()
        .to_rgb8()
        .save(&format!("{IMAGE_DIR_PATH}/{index}.bmp"))?;
    }

    Ok(())
}

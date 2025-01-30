#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines
)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use gl::types::{GLenum, GLfloat, GLint, GLuint};
use glutin::{
    config::{ConfigTemplateBuilder, GetGlConfig},
    context::{ContextApi, ContextAttributesBuilder, PossiblyCurrentContext},
    display::GetGlDisplay,
    prelude::{GlDisplay, NotCurrentGlContext},
    surface::{GlSurface, Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface},
};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;
use std::{
    ffi::{CStr, CString},
    fs,
    io::Cursor,
    num::NonZeroU32,
    path::Path,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    error::EventLoopError,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};
use zip::ZipArchive;

const IMAGE_ZIP_URL: &str = "http://m.adept.care/display_tv/MH.zip";
const IMAGE_ZIP_PATH: &str = "images.zip";
const IMAGE_DIR_PATH: &str = "images";

const TIME_BETWEEN_IMAGES: f64 = 3.0;
const TRANSITION_TIME: f64 = 1.0;

const WAIT_TIME: Duration = Duration::from_millis(66); // Frame time between refreshes

pub mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
    pub use Gles2 as Gl;
}

pub fn main() -> Result<(), EventLoopError> {
    // Grab new images on startup
    ensure_latest_images().unwrap();
    let image_paths = fs::read_dir(IMAGE_DIR_PATH)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|e| e.path().to_str().unwrap().to_owned()))
        .collect::<Vec<_>>();

    // Start the event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut App {
        template: ConfigTemplateBuilder::new(),
        gl_display: GlDisplayCreationState::Builder(
            DisplayBuilder::new().with_window_attributes(Some(window_attributes())),
        ),
        gl_context: None,
        state: None,
        renderer: None,
        image_paths,
        current_image_index: 0,
        start_time: Instant::now(),
    })
}

struct App {
    template: ConfigTemplateBuilder,
    renderer: Option<Renderer>,
    state: Option<AppState>,
    gl_context: Option<PossiblyCurrentContext>,
    gl_display: GlDisplayCreationState,
    image_paths: Vec<String>,
    current_image_index: usize,
    start_time: Instant,
}

struct AppState {
    gl_surface: Surface<WindowSurface>,
    window: Window,
}

enum GlDisplayCreationState {
    /// The display was not build yet.
    Builder(DisplayBuilder),
    /// The display was already created for the application.
    Init,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let (window, gl_config) = match &self.gl_display {
            GlDisplayCreationState::Builder(builder) => {
                let (window, config) = builder
                    .clone()
                    .build(event_loop, self.template.clone(), |mut configs| {
                        configs.next().unwrap()
                    })
                    .unwrap();
                self.gl_display = GlDisplayCreationState::Init;
                (window.unwrap(), config)
            }
            GlDisplayCreationState::Init => {
                let gl_config = self.gl_context.as_ref().unwrap().config();
                let window =
                    glutin_winit::finalize_window(event_loop, window_attributes(), &gl_config)
                        .unwrap();
                (window, gl_config)
            }
        };

        let attrs = window
            .build_surface_attributes(SurfaceAttributesBuilder::default())
            .unwrap();
        let gl_surface = unsafe {
            gl_config
                .display()
                .create_window_surface(&gl_config, &attrs)
                .unwrap()
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
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => (),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
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

        event_loop.set_control_flow(ControlFlow::WaitUntil(next_wake));
    }
}

fn window_attributes() -> WindowAttributes {
    Window::default_attributes()
        .with_title("Slideshow")
        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
}

pub struct Renderer {
    program: GLuint,
    vao: gl::types::GLuint,
    textures: [GLuint; 2],
    gl: gl::Gl,
    transition_location: GLint,
}

impl Renderer {
    pub fn new<D: GlDisplay>(gl_display: &D, image_paths: &[String], start_index: usize) -> Self {
        let gl = gl::Gl::load_with(|s| {
            gl_display
                .get_proc_address(CString::new(s).unwrap().as_c_str())
                .cast()
        });
        if let Some(renderer) = get_gl_string(&gl, gl::RENDERER) {
            println!("Running on {}", renderer.to_string_lossy());
        }
        if let Some(version) = get_gl_string(&gl, gl::VERSION) {
            println!("OpenGL Version {}", version.to_string_lossy());
        }
        if let Some(shaders_version) = get_gl_string(&gl, gl::SHADING_LANGUAGE_VERSION) {
            println!("Shaders version on {}", shaders_version.to_string_lossy());
        }

        // Create the shader program
        let program = unsafe {
            let program = gl.CreateProgram();
            let vertex_shader = compile_shader(&gl, gl::VERTEX_SHADER, VERTEX_SHADER);
            let fragment_shader = compile_shader(&gl, gl::FRAGMENT_SHADER, FRAGMENT_SHADER);
            gl.AttachShader(program, vertex_shader);
            gl.AttachShader(program, fragment_shader);
            gl.LinkProgram(program);
            gl.UseProgram(program);
            gl.DeleteShader(vertex_shader);
            gl.DeleteShader(fragment_shader);
            program
        };

        // Retrieve and set the uniform locations for the texture samplers
        unsafe {
            gl.Uniform1i(
                gl.GetUniformLocation(program, c"u_texture0".as_ptr().cast::<_>()),
                0,
            );
            gl.Uniform1i(
                gl.GetUniformLocation(program, c"u_texture1".as_ptr().cast::<_>()),
                1,
            );
        }

        // Create the vertex array object
        let vao = unsafe {
            let (mut vao, mut vbo) = (0, 0);
            gl.GenVertexArrays(1, &mut vao);
            gl.BindVertexArray(vao);
            gl.GenBuffers(1, &mut vbo);
            gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl.BufferData(
                gl::ARRAY_BUFFER,
                (VERTEX_DATA.len() * std::mem::size_of::<f32>()) as isize,
                VERTEX_DATA.as_ptr().cast(),
                gl::STATIC_DRAW,
            );

            let pos_attrib = gl.GetAttribLocation(program, c"position".as_ptr().cast());
            gl.VertexAttribPointer(pos_attrib as u32, 2, gl::FLOAT, 0, 16, std::ptr::null());
            gl.EnableVertexAttribArray(pos_attrib as u32);

            let tex_attrib = gl.GetAttribLocation(program, c"tex_coord".as_ptr().cast());
            gl.VertexAttribPointer(
                tex_attrib as u32,
                2,
                gl::FLOAT,
                0,
                16,
                (8 as *const ()).cast(),
            );
            gl.EnableVertexAttribArray(tex_attrib as u32);
            vao
        };

        // Retrieve the location of the transition uniform
        let transition_location =
            unsafe { gl.GetUniformLocation(program, c"transition".as_ptr().cast::<_>()) };

        // Load the textures
        let mut textures = [0, 0];
        unsafe {
            gl.GenTextures(2, textures.as_mut_ptr());
            gl.BindTexture(gl::TEXTURE_2D, textures[0]);
            load_texture_from_path(&gl, &image_paths[start_index]);
            gl.BindTexture(gl::TEXTURE_2D, textures[1]);
            load_texture_from_path(&gl, &image_paths[(start_index + 1) % image_paths.len()]);
        }

        Self {
            program,
            vao,
            textures,
            gl,
            transition_location,
        }
    }

    pub fn update_textures(&mut self, image_paths: &[String], current_index: usize) {
        unsafe {
            self.textures = (self.textures[1], self.textures[0]).into();

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

            self.gl.ActiveTexture(gl::TEXTURE0);
            self.gl.BindTexture(gl::TEXTURE_2D, self.textures[0]);

            self.gl.ActiveTexture(gl::TEXTURE1);
            self.gl.BindTexture(gl::TEXTURE_2D, self.textures[1]);

            self.gl
                .Uniform1f(self.transition_location, transition_value);

            self.gl.ClearColor(0.0, 0.0, 0.0, 1.0);
            self.gl.Clear(gl::COLOR_BUFFER_BIT);
            self.gl.DrawArrays(gl::TRIANGLES, 0, 6);
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

unsafe fn compile_shader(gl: &gl::Gl, ty: u32, source: &[u8]) -> GLuint {
    let shader = gl.CreateShader(ty);
    gl.ShaderSource(
        shader,
        1,
        [source.as_ptr().cast()].as_ptr(),
        std::ptr::null(),
    );
    gl.CompileShader(shader);
    shader
}

fn get_gl_string(gl: &gl::Gl, variant: GLenum) -> Option<&'static CStr> {
    unsafe {
        let s = gl.GetString(variant);
        (!s.is_null()).then(|| CStr::from_ptr(s.cast()))
    }
}

#[rustfmt::skip]
static VERTEX_DATA: [f32; 24] = [
    -1.0, -1.0, 0.0, 0.0,
     1.0, -1.0, 1.0, 0.0,
     1.0,  1.0, 1.0, 1.0,
    -1.0, -1.0, 0.0, 0.0,
     1.0,  1.0, 1.0, 1.0,
    -1.0,  1.0, 0.0, 1.0,
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
/// Otherwise, it downloads the latest images from `IMAGE_ZIP_URL`, unpacks them to `IMAGE_DIR_PATH`
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

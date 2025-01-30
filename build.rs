use gl_generator::{Api, Fallbacks, Profile, Registry, StructGenerator};
use std::{env, fs::File, path::PathBuf};

fn main() {
    // Define the output path for generated bindings.
    let dest = PathBuf::from(&env::var("OUT_DIR").unwrap());

    // Create the bindings file.
    let mut file = File::create(dest.join("gl_bindings.rs")).unwrap();

    // Generate OpenGL ES bindings.
    Registry::new(Api::Gles2, (3, 0), Profile::Core, Fallbacks::All, [])
        .write_bindings(StructGenerator, &mut file)
        .unwrap();

    println!("cargo:rerun-if-changed=build.rs");
}

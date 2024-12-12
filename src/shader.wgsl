struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),  // Bottom-left
        vec2<f32>( 1.0, -1.0),  // Bottom-right
        vec2<f32>(-1.0,  1.0),  // Top-left
        vec2<f32>( 1.0,  1.0)   // Top-right
    );

    var tex_coords = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0)
    );

    var output: VertexOutput;
    output.pos = vec4<f32>(positions[input.vertex_index], 0.0, 1.0);
    output.uv = tex_coords[input.vertex_index];
    return output;
}

@group(0) @binding(0) var front_texture: texture_2d<f32>;
@group(0) @binding(1) var front_sampler: sampler;
@group(0) @binding(2) var back_texture: texture_2d<f32>;
@group(0) @binding(3) var back_sampler: sampler;
@group(0) @binding(4) var<uniform> mix_factor: f32;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample colors from both textures
    let front_color = textureSample(front_texture, front_sampler, input.uv);
    let back_color = textureSample(back_texture, back_sampler, input.uv);
    
    // Mix colors based on mix_factor for fading effect
    return mix(front_color, back_color, mix_factor);
}
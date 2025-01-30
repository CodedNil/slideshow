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
        vec2<f32>(1.0, -1.0),  // Bottom-right
        vec2<f32>(-1.0, 1.0),  // Top-left
        vec2<f32>(1.0, 1.0)   // Top-right
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

@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var front_texture: texture_2d<f32>;
@group(0) @binding(2) var back_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> mix_factor: f32;

struct PushConstants {
    mix_factor: f32,
};
var<push_constant> push_constants: PushConstants;


@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let front_color = textureSample(front_texture, texture_sampler, input.uv);
    let back_color = textureSample(back_texture, texture_sampler, input.uv);
    let mix_factor = push_constants.mix_factor;

    // Directly return based on mix factor for simple cases
    if mix_factor <= 0.0 {
        return front_color;
    }
    if mix_factor >= 1.0 {
        return back_color;
    }

    // Mix colors based on mix_factor
    return mix(front_color, back_color, mix_factor);
}
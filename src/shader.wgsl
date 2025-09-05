// 顶点着色器输入
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

// 顶点着色器输出
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

// 相机uniform
@group(0) @binding(0)
var<uniform> camera: mat4x4<f32>;

// 顶点着色器
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = input.position;
    out.normal = input.normal;
    out.color = input.color;
    out.clip_position = camera * vec4<f32>(input.position, 1.0);
    return out;
}

// 片段着色器
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // 简单的光照计算
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let normal = normalize(input.normal);
    
    // 环境光
    let ambient = 0.3;
    
    // 漫反射光
    let diffuse = max(dot(normal, light_dir), 0.0) * 0.7;
    
    // 最终颜色
    let lighting = ambient + diffuse;
    let final_color = input.color.rgb * lighting;
    
    return vec4<f32>(final_color, input.color.a);
}

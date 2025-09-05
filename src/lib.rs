use std::sync::Arc;

use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::PhysicalKey,
    window::Window,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod camera;
pub mod config;
mod geometry;
mod pdb_parser;

use camera::{Camera, CameraController, CameraUniform};
use config::AppConfig;
use geometry::{ProteinRenderer, Vertex};
use pdb_parser::Protein;

// This will store the state of our protein viewer
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    line_render_pipeline: wgpu::RenderPipeline, // 用于渲染坐标轴线条
    transparent_render_pipeline: wgpu::RenderPipeline, // 用于渲染透明表面
    window: Arc<Window>,

    // 配置
    app_config: AppConfig,

    // 蛋白质相关
    proteins: Vec<Protein>,
    protein_renderers: Vec<ProteinRenderer>,
    vertex_buffers: Vec<Option<wgpu::Buffer>>,
    index_buffers: Vec<Option<wgpu::Buffer>>,
    num_indices_list: Vec<u32>,
    buffer_styles: Vec<geometry::RepresentationStyle>, // 跟踪每个缓冲区的样式

    // 相机相关
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // 深度缓冲
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // 坐标轴
    axes_vertex_buffer: wgpu::Buffer,
    axes_index_buffer: wgpu::Buffer,
    axes_num_indices: u32,
    axes_camera_uniform: CameraUniform,
    axes_camera_buffer: wgpu::Buffer,
    axes_camera_bind_group: wgpu::BindGroup,
}

impl State {
    pub async fn new(window: Arc<Window>, app_config: AppConfig) -> anyhow::Result<Self> {
        let size = window.inner_size();

        // 确保尺寸至少是1x1，避免零尺寸错误
        let width = if size.width == 0 { 800 } else { size.width };
        let height = if size.height == 0 { 600 } else { size.height };

        log::info!("窗口尺寸: {}x{}", width, height);

        // The instance is a handle to our GPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width,
            height: height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // 创建深度纹理
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 初始化相机，使用配置文件设置
        let camera = Camera::new(
            (
                app_config.camera.position[0],
                app_config.camera.position[1],
                app_config.camera.position[2],
            ),
            (
                app_config.camera.target[0],
                app_config.camera.target[1],
                app_config.camera.target[2],
            ),
            cgmath::Vector3::new(
                app_config.camera.up[0],
                app_config.camera.up[1],
                app_config.camera.up[2],
            ),
            config.width as f32 / config.height as f32,
            cgmath::Deg(app_config.camera.fov_degrees),
            app_config.camera.near,
            app_config.camera.far,
        );

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let camera_controller = CameraController::new(
            app_config.camera.move_speed,
            app_config.camera.rotation_speed,
        );

        // 创建着色器和渲染管线
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // 创建线条渲染管道（用于坐标轴）
        let line_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("线条渲染管道"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[geometry::Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 改为三角形列表
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back), // 启用背面剔除
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // 创建透明渲染管道（禁用深度写入）
        let transparent_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("透明渲染管道"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[geometry::Vertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false, // 禁用深度写入
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        // 创建坐标轴几何体
        let axes = geometry::CoordinateAxes::new_rotating_axes();
        let axes_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("坐标轴顶点缓冲区"),
            contents: bytemuck::cast_slice(&axes.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let axes_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("坐标轴索引缓冲区"),
            contents: bytemuck::cast_slice(&axes.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let axes_num_indices = axes.indices.len() as u32;

        // 创建坐标轴专用相机
        let mut axes_camera_uniform = CameraUniform::new();

        // 初始化坐标轴的投影矩阵（将在update中更新）
        axes_camera_uniform.view_proj = cgmath::Matrix4::identity().into();

        let axes_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("坐标轴相机缓冲区"),
            contents: bytemuck::cast_slice(&[axes_camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let axes_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: axes_camera_buffer.as_entire_binding(),
            }],
            label: Some("坐标轴相机绑定组"),
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            line_render_pipeline,
            transparent_render_pipeline,
            window,
            app_config,
            proteins: Vec::new(),
            protein_renderers: Vec::new(),
            vertex_buffers: Vec::new(),
            index_buffers: Vec::new(),
            num_indices_list: Vec::new(),
            buffer_styles: Vec::new(),
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            depth_view,
            axes_vertex_buffer,
            axes_index_buffer,
            axes_num_indices,
            axes_camera_uniform,
            axes_camera_buffer,
            axes_camera_bind_group,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            // 更新深度纹理
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth_texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.depth_view = self
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            // 更新相机纵横比
            self.camera.aspect = width as f32 / height as f32;
        }
    }

    pub fn load_proteins_from_config(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!("加载蛋白质...");

            let protein_configs = self.app_config.proteins.clone();
            for protein_config in &protein_configs {
                match crate::pdb_parser::load_pdb_file(protein_config.path.to_str().unwrap()) {
                    Ok(protein) => {
                        println!("成功加载蛋白质: {:?}", protein_config.path);
                        self.add_protein(protein, protein_config.clone());
                    }
                    Err(e) => {
                        println!("加载蛋白质失败 {:?}: {}", protein_config.path, e);
                    }
                }
            }

            // 如果加载了蛋白质，设置相机
            if !self.proteins.is_empty() {
                self.setup_camera_for_proteins();
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            println!("WASM环境：准备等待用户加载蛋白质文件...");
            // 在WASM中，我们不会立即加载蛋白质
            // 而是等待用户通过文件上传或URL输入来加载
            // 可以设置一个默认相机位置
            self.camera.eye = cgmath::Point3::new(0.0, 0.0, 10.0);
            self.camera.target = cgmath::Point3::new(0.0, 0.0, 0.0);
            self.camera.up = cgmath::Vector3::unit_y();
        }
    }

    pub fn add_protein(&mut self, protein: Protein, config: config::ProteinConfig) {
        let protein_index = self.proteins.len();

        // 计算并显示信息
        let center = protein.center();
        println!(
            "蛋白质 {} 中心: ({:.2}, {:.2}, {:.2})",
            protein_index, center[0], center[1], center[2]
        );

        // 添加蛋白质
        self.proteins.push(protein);

        // 为每种渲染样式创建渲染器
        for style in &config.styles {
            let mut renderer = ProteinRenderer::new();
            renderer.style = match style {
                config::RepresentationStyle::SpaceFilling => {
                    geometry::RepresentationStyle::SpaceFilling
                }
                config::RepresentationStyle::BallAndStick => {
                    geometry::RepresentationStyle::BallAndStick
                }
                config::RepresentationStyle::MolecularSurface => {
                    geometry::RepresentationStyle::MolecularSurface
                }
                config::RepresentationStyle::Backbone => geometry::RepresentationStyle::Backbone,
                config::RepresentationStyle::Cartoon => geometry::RepresentationStyle::Cartoon,
            };
            renderer.show_hydrogens = config.show_hydrogens;

            // 更新几何体
            if let Some(protein) = self.proteins.last() {
                renderer.update_geometry(protein);
            }

            // 创建缓冲区
            let (vertex_buffer, index_buffer, num_indices) =
                if !renderer.geometry.vertices.is_empty() {
                    let vb = Some(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("Vertex Buffer {}", protein_index)),
                            contents: bytemuck::cast_slice(&renderer.geometry.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));

                    let ib = Some(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("Index Buffer {}", protein_index)),
                            contents: bytemuck::cast_slice(&renderer.geometry.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        },
                    ));

                    let ni = renderer.geometry.indices.len() as u32;
                    println!(
                        "蛋白质 {} 样式 {:?}: {} 顶点, {} 索引",
                        protein_index,
                        style,
                        renderer.geometry.vertices.len(),
                        ni
                    );

                    (vb, ib, ni)
                } else {
                    (None, None, 0)
                };

            let style = renderer.style.clone();
            self.protein_renderers.push(renderer);
            self.vertex_buffers.push(vertex_buffer);
            self.index_buffers.push(index_buffer);
            self.num_indices_list.push(num_indices);
            self.buffer_styles.push(style);
        }
    }

    fn setup_camera_for_proteins(&mut self) {
        if self.proteins.is_empty() {
            return;
        }

        // 计算所有蛋白质的包围盒
        let mut min_bounds = [f32::INFINITY; 3];
        let mut max_bounds = [f32::NEG_INFINITY; 3];

        for protein in &self.proteins {
            for atom in &protein.atoms {
                let radius = atom.atom_type.radius();
                min_bounds[0] = min_bounds[0].min(atom.x - radius);
                min_bounds[1] = min_bounds[1].min(atom.y - radius);
                min_bounds[2] = min_bounds[2].min(atom.z - radius);
                max_bounds[0] = max_bounds[0].max(atom.x + radius);
                max_bounds[1] = max_bounds[1].max(atom.y + radius);
                max_bounds[2] = max_bounds[2].max(atom.z + radius);
            }
        }

        // 计算中心和半径
        let center = [
            (min_bounds[0] + max_bounds[0]) / 2.0,
            (min_bounds[1] + max_bounds[1]) / 2.0,
            (min_bounds[2] + max_bounds[2]) / 2.0,
        ];

        let radius = ((max_bounds[0] - min_bounds[0]).powi(2)
            + (max_bounds[1] - min_bounds[1]).powi(2)
            + (max_bounds[2] - min_bounds[2]).powi(2))
        .sqrt()
            / 2.0;

        println!(
            "所有蛋白质的包围球 - 中心: ({:.2}, {:.2}, {:.2}), 半径: {:.2}",
            center[0], center[1], center[2], radius
        );

        // 设置相机目标和距离
        self.camera_controller
            .set_target(cgmath::Point3::new(center[0], center[1], center[2]));

        let fovy_rad = self.camera.fovy;
        let aspect = self.camera.aspect;
        let vertical_half_angle = fovy_rad / 2.0;
        let horizontal_half_angle = (vertical_half_angle.tan() * aspect).atan();
        let max_half_angle = vertical_half_angle.max(horizontal_half_angle);
        let min_distance = radius / max_half_angle.sin() + 10.0;

        self.camera_controller.set_distance(min_distance);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        // We can't render unless the surface is configured
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // 第一阶段：渲染不透明物体
            for i in 0..self.vertex_buffers.len() {
                if let (Some(vb), Some(ib)) = (&self.vertex_buffers[i], &self.index_buffers[i]) {
                    let num_indices = self.num_indices_list[i];
                    let style = &self.buffer_styles[i];

                    // 只渲染不透明物体
                    if !matches!(style, geometry::RepresentationStyle::MolecularSurface)
                        && num_indices > 0
                    {
                        render_pass.set_vertex_buffer(0, vb.slice(..));
                        render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..num_indices, 0, 0..1);
                    }
                }
            }

            // 第二阶段：渲染透明物体（使用透明管道）
            render_pass.set_pipeline(&self.transparent_render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            for i in 0..self.vertex_buffers.len() {
                if let (Some(vb), Some(ib)) = (&self.vertex_buffers[i], &self.index_buffers[i]) {
                    let num_indices = self.num_indices_list[i];
                    let style = &self.buffer_styles[i];

                    // 只渲染透明物体（分子表面）
                    if matches!(style, geometry::RepresentationStyle::MolecularSurface)
                        && num_indices > 0
                    {
                        render_pass.set_vertex_buffer(0, vb.slice(..));
                        render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..num_indices, 0, 0..1);
                    }
                }
            }

            // 渲染坐标轴
            if self.app_config.render.show_axes {
                render_pass.set_pipeline(&self.line_render_pipeline);
                render_pass.set_bind_group(0, &self.axes_camera_bind_group, &[]); // 使用坐标轴专用相机
                render_pass.set_vertex_buffer(0, self.axes_vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.axes_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.axes_num_indices, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update(&mut self) {
        // 更新相机控制器
        self.camera_controller.update_camera(&mut self.camera);

        // 更新相机uniform
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // 更新坐标轴相机，让它跟随主相机旋转但固定在屏幕左下角
        self.update_axes_camera();
    }

    fn update_axes_camera(&mut self) {
        // 获取相机的旋转部分（不包含平移）
        let view_matrix =
            cgmath::Matrix4::look_at_rh(self.camera.eye, self.camera.target, self.camera.up);

        // 提取旋转矩阵（移除平移）
        let rotation_matrix = cgmath::Matrix4::new(
            view_matrix.x.x,
            view_matrix.x.y,
            view_matrix.x.z,
            0.0,
            view_matrix.y.x,
            view_matrix.y.y,
            view_matrix.y.z,
            0.0,
            view_matrix.z.x,
            view_matrix.z.y,
            view_matrix.z.z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        // 创建正交投影矩阵
        let ortho_matrix = cgmath::ortho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

        // 缩放和平移矩阵，将坐标轴放置在屏幕左下角
        let scale_translate =
            cgmath::Matrix4::from_translation(cgmath::Vector3::new(-0.7, -0.7, 0.0))
                * cgmath::Matrix4::from_scale(0.2);

        // 组合变换：先应用旋转，然后缩放和平移
        let final_matrix =
            camera::OPENGL_TO_WGPU_MATRIX * ortho_matrix * scale_translate * rotation_matrix;

        self.axes_camera_uniform.view_proj = final_matrix.into();

        // 更新坐标轴相机缓冲区
        self.queue.write_buffer(
            &self.axes_camera_buffer,
            0,
            bytemuck::cast_slice(&[self.axes_camera_uniform]),
        );
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
    config: AppConfig,
}

impl App {
    pub fn new(config: AppConfig) -> Self {
        Self {
            state: None,
            config,
            #[cfg(target_arch = "wasm32")]
            proxy: None,
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_with_proxy(config: AppConfig, proxy: EventLoopProxy<State>) -> Self {
        Self {
            state: None,
            config,
            proxy: Some(proxy),
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes()
            .with_title(&self.config.render.window_title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.render.window_size.0,
                self.config.render.window_size.1,
            ));

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "wgpu-canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            // If we are not on web we can use pollster to
            // await the
            let mut state = pollster::block_on(State::new(window, self.config.clone())).unwrap();
            state.load_proteins_from_config();
            self.state = Some(state);
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Run the future asynchronously and use the
            // proxy to send the results to the event loop
            if let Some(proxy) = self.proxy.take() {
                let config = self.config.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(
                                State::new(window, config)
                                    .await
                                    .expect("Unable to create canvas!!!")
                            )
                            .is_ok()
                    )
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            log::info!("收到State事件，开始设置全局状态");

            // 加载蛋白质数据
            event.load_proteins_from_config();

            // 请求重绘以显示蛋白质
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );

            // 设置全局状态
            let state_rc = Rc::new(RefCell::new(event));
            set_global_state(state_rc.clone());

            // 启动渲染循环
            start_render_loop();

            log::info!("全局状态设置完成，蛋白质数据已加载");
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(event);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        #[cfg(target_arch = "wasm32")]
        {
            // 在WASM中，使用全局状态
            let _ = with_global_state(|state| {
                // 先让相机控制器处理事件
                if state.camera_controller.process_events(&event) {
                    return;
                }

                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Resized(size) => state.resize(size.width, size.height),
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            // Reconfigure the surface if it's lost or outdated
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let size = state.window.inner_size();
                                state.resize(size.width, size.height);
                            }
                            Err(e) => {
                                log::error!("Unable to render {}", e);
                            }
                        }
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(code),
                                state: _key_state,
                                ..
                            },
                        ..
                    } => {
                        // 移除旧的键盘处理，将在后续版本中重新实现
                        println!("键盘输入: {:?}", code);
                    }
                    _ => {}
                }
            });
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = match &mut self.state {
                Some(canvas) => canvas,
                None => return,
            };

            // 先让相机控制器处理事件
            if state.camera_controller.process_events(&event) {
                return;
            }

            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => state.resize(size.width, size.height),
                WindowEvent::RedrawRequested => {
                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        // Reconfigure the surface if it's lost or outdated
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = state.window.inner_size();
                            state.resize(size.width, size.height);
                        }
                        Err(e) => {
                            log::error!("Unable to render {}", e);
                        }
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(code),
                            state: _key_state,
                            ..
                        },
                    ..
                } => {
                    // 移除旧的键盘处理，将在后续版本中重新实现
                    println!("键盘输入: {:?}", code);
                }
                _ => {}
            }
        }
    }
}

pub fn run_with_config(config: AppConfig) -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;

    Ok(())
}

pub fn run() -> anyhow::Result<()> {
    run_with_config(AppConfig::default())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn run_app() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap_throw();

    log::info!("开始初始化WASM应用");

    let config = AppConfig::default();
    let event_loop = EventLoop::with_user_event().build().map_err(|e| {
        log::error!("创建事件循环失败: {:?}", e);
        wasm_bindgen::JsValue::from_str(&format!("创建事件循环失败: {:?}", e))
    })?;

    let proxy = event_loop.create_proxy();
    let mut app = App::new_with_proxy(config, proxy);

    // 在WASM中，run_app不会返回，它会接管控制权
    // 我们需要处理这个错误，但不能阻止执行
    let _ = event_loop.run_app(&mut app);

    Ok(())
}

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
fn set_global_state(state: Rc<RefCell<State>>) {
    GLOBAL_STATE.with(|global_state| {
        *global_state.borrow_mut() = Some(state);
    });
}

#[cfg(target_arch = "wasm32")]
fn with_global_state<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut State) -> R,
{
    GLOBAL_STATE.with(|global_state| {
        if let Some(state_rc) = global_state.borrow().as_ref() {
            let mut state = state_rc.borrow_mut();
            Some(f(&mut *state))
        } else {
            None
        }
    })
}

#[cfg(target_arch = "wasm32")]
fn start_render_loop() {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    type RenderClosure = Closure<dyn FnMut()>;
    let f: Rc<RefCell<Option<RenderClosure>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // 进行渲染
        let _ = with_global_state(|state| {
            state.update();
            if let Err(e) = state.render() {
                log::error!("渲染错误: {:?}", e);
            }
        });

        // 请求下一帧
        if let Some(window) = web_sys::window() {
            let _ = window
                .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref());
        }
    }) as Box<dyn FnMut()>));

    // 启动第一帧
    if let Some(window) = web_sys::window() {
        let _ =
            window.request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref());
    }
}

/// 从URL加载PDB文件并添加到场景中
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn load_protein_from_url(url: &str) -> Result<(), wasm_bindgen::JsValue> {
    use crate::config::ProteinConfig;
    use crate::pdb_parser::load_pdb_from_url;
    use std::path::PathBuf;

    let protein = load_pdb_from_url(url).await?;
    log::info!("成功从URL加载蛋白质: {} 个原子", protein.atoms.len());

    // 创建默认配置
    let config = ProteinConfig {
        path: PathBuf::from(url),
        styles: vec![crate::config::RepresentationStyle::BallAndStick],
        show_hydrogens: false,
        color_scheme: None,
        opacity: 1.0,
        visible: true,
    };

    // 将蛋白质添加到全局状态
    if let Some(_) = with_global_state(|state| {
        state.add_protein(protein, config);
        state.setup_camera_for_proteins();
    }) {
        log::info!("蛋白质已成功添加到场景中");
    } else {
        return Err(wasm_bindgen::JsValue::from_str("无法访问全局状态"));
    }

    Ok(())
}

/// 从Blob加载PDB文件并添加到场景中
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn load_protein_from_blob(blob: &web_sys::Blob) -> Result<(), wasm_bindgen::JsValue> {
    use crate::config::ProteinConfig;
    use crate::pdb_parser::load_pdb_from_blob;
    use std::path::PathBuf;

    let protein = load_pdb_from_blob(blob).await?;
    log::info!("成功从Blob加载蛋白质: {} 个原子", protein.atoms.len());

    // 创建默认配置
    let config = ProteinConfig {
        path: PathBuf::from("blob"),
        styles: vec![crate::config::RepresentationStyle::BallAndStick],
        show_hydrogens: false,
        color_scheme: None,
        opacity: 1.0,
        visible: true,
    };

    // 将蛋白质添加到全局状态
    if let Some(_) = with_global_state(|state| {
        state.add_protein(protein, config);
        state.setup_camera_for_proteins();
    }) {
        log::info!("蛋白质已成功添加到场景中");
    } else {
        return Err(wasm_bindgen::JsValue::from_str("无法访问全局状态"));
    }

    Ok(())
}

/// 从File对象加载PDB文件并添加到场景中
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn load_protein_from_file(file: &web_sys::File) -> Result<(), wasm_bindgen::JsValue> {
    use crate::config::ProteinConfig;
    use crate::pdb_parser::load_pdb_from_file;
    use std::path::PathBuf;

    let protein = load_pdb_from_file(file).await?;
    let filename = file.name();
    log::info!(
        "成功从文件 {} 加载蛋白质: {} 个原子",
        filename,
        protein.atoms.len()
    );

    // 创建默认配置
    let config = ProteinConfig {
        path: PathBuf::from(&filename),
        styles: vec![crate::config::RepresentationStyle::BallAndStick],
        show_hydrogens: false,
        color_scheme: None,
        opacity: 1.0,
        visible: true,
    };

    // 将蛋白质添加到全局状态
    if let Some(_) = with_global_state(|state| {
        state.add_protein(protein, config);
        state.setup_camera_for_proteins();
    }) {
        log::info!("蛋白质已成功添加到场景中");
    } else {
        return Err(wasm_bindgen::JsValue::from_str("无法访问全局状态"));
    }

    Ok(())
}

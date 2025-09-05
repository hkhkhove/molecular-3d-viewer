use crate::pdb_parser::Protein;
use cgmath::{InnerSpace, Point3, Vector3};
use mcubes::{MarchingCubes, MeshSide};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3], //法线向量
    pub color: [f32; 4],  // 改为RGBA，包含透明度
}

impl Vertex {
    //向 wgpu 渲染管线描述这个顶点数据的内存布局，告诉 GPU 如何解析顶点缓冲区
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            //每个顶点在缓冲区中占用的字节数（步长）
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            //顶点数据的步进模式
            step_mode: wgpu::VertexStepMode::Vertex, //每个顶点推进一次
            attributes: &[
                //position 12字节
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0, //对应着色器中的@location(0)
                    format: wgpu::VertexFormat::Float32x3,
                },
                //normal 法向量 12字节
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                //color 16字节 (RGBA)
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

//通用的容器，用于存储任何一个 3D 物体的完整几何信息
pub struct GeometryData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl GeometryData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(), //顶点的索引，每三个连接成一个三角形
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    //多个几何体合并成一个
    pub fn extend(&mut self, other: GeometryData) {
        let vertex_offset = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        self.indices
            .extend(other.indices.iter().map(|i| i + vertex_offset));
    }
}

pub struct CoordinateAxes {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl CoordinateAxes {
    // 创建带箭头的跟随旋转坐标轴
    pub fn new_rotating_axes() -> Self {
        let mut geometry = GeometryData::new();

        let axis_length = 0.8;
        let axis_thickness = 0.02; // 轴的粗细
        let arrow_length = 0.2;
        let arrow_radius = 0.06;

        // X轴（红色）- 从原点指向右侧（标准右手坐标系）
        let x_axis_start = Point3::new(0.0, 0.0, 0.0);
        let x_axis_end = Point3::new(axis_length - arrow_length, 0.0, 0.0);
        let x_axis_tip = Point3::new(axis_length, 0.0, 0.0);

        // X轴主体圆柱体
        let x_cylinder = create_cylinder_geometry(
            x_axis_start,
            x_axis_end,
            axis_thickness,
            [1.0, 0.0, 0.0, 1.0],
            12,
        );
        geometry.extend(x_cylinder);

        // X轴箭头锥体
        let x_arrow = create_cone_geometry(
            x_axis_end,
            x_axis_tip,
            arrow_radius,
            [1.0, 0.0, 0.0, 1.0],
            12,
        );
        geometry.extend(x_arrow);

        // Y轴（绿色）- 从原点指向上方
        let y_axis_start = Point3::new(0.0, 0.0, 0.0);
        let y_axis_end = Point3::new(0.0, axis_length - arrow_length, 0.0);
        let y_axis_tip = Point3::new(0.0, axis_length, 0.0);

        // Y轴主体圆柱体
        let y_cylinder = create_cylinder_geometry(
            y_axis_start,
            y_axis_end,
            axis_thickness,
            [0.0, 1.0, 0.0, 1.0],
            12,
        );
        geometry.extend(y_cylinder);

        // Y轴箭头锥体
        let y_arrow = create_cone_geometry(
            y_axis_end,
            y_axis_tip,
            arrow_radius,
            [0.0, 1.0, 0.0, 1.0],
            12,
        );
        geometry.extend(y_arrow);

        // Z轴（蓝色）- 从原点指向观察者（屏幕外）
        let z_axis_start = Point3::new(0.0, 0.0, 0.0);
        let z_axis_end = Point3::new(0.0, 0.0, axis_length - arrow_length);
        let z_axis_tip = Point3::new(0.0, 0.0, axis_length);

        // Z轴主体圆柱体
        let z_cylinder = create_cylinder_geometry(
            z_axis_start,
            z_axis_end,
            axis_thickness,
            [0.0, 0.0, 1.0, 1.0],
            12,
        );
        geometry.extend(z_cylinder);

        // Z轴箭头锥体
        let z_arrow = create_cone_geometry(
            z_axis_end,
            z_axis_tip,
            arrow_radius,
            [0.0, 0.0, 1.0, 1.0],
            12,
        );
        geometry.extend(z_arrow);

        Self {
            vertices: geometry.vertices,
            indices: geometry.indices,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RepresentationStyle {
    SpaceFilling,
    BallAndStick,
    MolecularSurface,
    Backbone,
    Cartoon,
}

pub struct ProteinRenderer {
    pub style: RepresentationStyle,
    pub show_hydrogens: bool,
    pub geometry: GeometryData,
}

impl ProteinRenderer {
    pub fn new() -> Self {
        Self {
            style: RepresentationStyle::BallAndStick,
            show_hydrogens: false,
            geometry: GeometryData::new(),
        }
    }

    pub fn update_geometry(&mut self, protein: &Protein) {
        self.geometry.clear();

        let filtered_atoms: Vec<_> = if self.show_hydrogens {
            protein.atoms.iter().collect()
        } else {
            protein
                .atoms
                .iter()
                .filter(|atom| atom.atom_type != crate::pdb_parser::AtomType::Hydrogen)
                .collect()
        };

        match self.style {
            RepresentationStyle::SpaceFilling => {
                for atom in &filtered_atoms {
                    let radius = atom.atom_type.radius();
                    let color = atom.atom_type.color();
                    let center = Point3::new(atom.x, atom.y, atom.z);

                    let sphere = create_sphere_geometry(center, radius, color, 20);
                    self.geometry.extend(sphere);
                }
            }
            RepresentationStyle::BallAndStick => {
                // 原子
                for atom in &filtered_atoms {
                    let radius = atom.atom_type.radius() * 0.3; // 原子球半径
                    let color = atom.atom_type.color();
                    let center = Point3::new(atom.x, atom.y, atom.z);

                    let sphere = create_sphere_geometry(center, radius, color, 8);
                    self.geometry.extend(sphere);
                }

                // 化学键
                for bond in &protein.bonds {
                    if let (Some(atom1), Some(atom2)) = (
                        protein.atoms.get(bond.atom1_index),
                        protein.atoms.get(bond.atom2_index),
                    ) {
                        // 如果不显示氢原子，跳过包含氢原子的键
                        if !self.show_hydrogens
                            && (atom1.atom_type == crate::pdb_parser::AtomType::Hydrogen
                                || atom2.atom_type == crate::pdb_parser::AtomType::Hydrogen)
                        {
                            continue;
                        }

                        let start = Point3::new(atom1.x, atom1.y, atom1.z);
                        let end = Point3::new(atom2.x, atom2.y, atom2.z);
                        let radius = 0.1; // 键的半径
                        let color = [0.7, 0.7, 0.7, 1.0];

                        let cylinder = create_cylinder_geometry(start, end, radius, color, 16);
                        self.geometry.extend(cylinder);
                    }
                }
            }
            RepresentationStyle::MolecularSurface => {
                // 生成分子表面
                let grid_resolution = 32; // 网格分辨率
                let probe_radius = 1.4; // 探针半径（水分子半径）
                let surface_geometry =
                    create_molecular_surface_geometry(protein, grid_resolution, probe_radius);
                self.geometry.extend(surface_geometry);
            }
            _ => {
                // 其他样式待实现
                log::warn!("Representation style not yet implemented");
            }
        }
    }
}

//球体
pub fn create_sphere_geometry(
    center: Point3<f32>,
    radius: f32,
    color: [f32; 4],
    segments: u32,
) -> GeometryData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let rings = segments;
    let sectors = segments * 2;

    // 生成顶点
    for ring in 0..=rings {
        let phi = std::f32::consts::PI * ring as f32 / rings as f32;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        for sector in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * sector as f32 / sectors as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            let x = sin_phi * cos_theta;
            let y = cos_phi;
            let z = sin_phi * sin_theta;

            let position = [
                center.x + radius * x,
                center.y + radius * y,
                center.z + radius * z,
            ];

            let normal = [x, y, z];

            vertices.push(Vertex {
                position,
                normal,
                color,
            });
        }
    }

    // 生成索引
    for ring in 0..rings {
        for sector in 0..sectors {
            let current = ring * (sectors + 1) + sector;
            let next = current + sectors + 1;

            indices.push(current);
            indices.push(next);
            indices.push(current + 1);

            indices.push(current + 1);
            indices.push(next);
            indices.push(next + 1);
        }
    }

    GeometryData { vertices, indices }
}

//圆柱体
pub fn create_cylinder_geometry(
    start: Point3<f32>,
    end: Point3<f32>,
    radius: f32,
    color: [f32; 4],
    segments: u32,
) -> GeometryData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let direction = Vector3::new(end.x - start.x, end.y - start.y, end.z - start.z);
    let length = direction.magnitude();
    let normalized_dir = direction / length;

    // 计算垂直于方向的两个向量
    let up = if normalized_dir.y.abs() < 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };

    let right = normalized_dir.cross(up).normalize();
    let forward = right.cross(normalized_dir).normalize();

    // 生成圆柱体顶点
    for ring in 0..=1 {
        let z = ring as f32 * length;

        for sector in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * sector as f32 / segments as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let local_pos = right * (radius * cos_theta) + forward * (radius * sin_theta);
            let position = [
                start.x + normalized_dir.x * z + local_pos.x,
                start.y + normalized_dir.y * z + local_pos.y,
                start.z + normalized_dir.z * z + local_pos.z,
            ];

            let normal = [
                local_pos.x / radius,
                local_pos.y / radius,
                local_pos.z / radius,
            ];

            vertices.push(Vertex {
                position,
                normal,
                color,
            });
        }
    }

    // 生成索引
    for sector in 0..segments {
        let current_bottom = sector;
        let next_bottom = current_bottom + 1;
        let current_top = sector + segments + 1;
        let next_top = current_top + 1;

        // 侧面四边形
        indices.push(current_bottom);
        indices.push(current_top);
        indices.push(next_bottom);

        indices.push(next_bottom);
        indices.push(current_top);
        indices.push(next_top);
    }

    GeometryData { vertices, indices }
}

//圆锥体
pub fn create_cone_geometry(
    base_center: Point3<f32>,
    tip: Point3<f32>,
    radius: f32,
    color: [f32; 4],
    segments: u32,
) -> GeometryData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let direction = Vector3::new(
        tip.x - base_center.x,
        tip.y - base_center.y,
        tip.z - base_center.z,
    );
    let length = direction.magnitude();
    let normalized_dir = direction / length;

    // 计算垂直于方向的两个向量
    let up = if normalized_dir.y.abs() < 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };

    let right = normalized_dir.cross(up).normalize();
    let forward = right.cross(normalized_dir).normalize();

    // 锥体顶点（尖端）
    vertices.push(Vertex {
        position: [tip.x, tip.y, tip.z],
        normal: normalized_dir.into(),
        color,
    });

    // 底面圆周上的顶点
    for sector in 0..segments {
        let theta = 2.0 * std::f32::consts::PI * sector as f32 / segments as f32;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let local_pos = right * (radius * cos_theta) + forward * (radius * sin_theta);
        let position = [
            base_center.x + local_pos.x,
            base_center.y + local_pos.y,
            base_center.z + local_pos.z,
        ];

        // 计算法向量（从圆心指向表面的方向）
        let surface_normal = (local_pos + normalized_dir * radius).normalize();
        let normal = [surface_normal.x, surface_normal.y, surface_normal.z];

        vertices.push(Vertex {
            position,
            normal,
            color,
        });
    }

    // 底面中心点
    vertices.push(Vertex {
        position: [base_center.x, base_center.y, base_center.z],
        normal: [-normalized_dir.x, -normalized_dir.y, -normalized_dir.z],
        color,
    });

    // 锥体侧面三角形
    for sector in 0..segments {
        let next_sector = (sector + 1) % segments;
        indices.push(0); // 顶点
        indices.push(1 + sector);
        indices.push(1 + next_sector);
    }

    // 底面三角形
    let base_center_index = segments + 1;
    for sector in 0..segments {
        let next_sector = (sector + 1) % segments;
        indices.push(base_center_index);
        indices.push(1 + next_sector);
        indices.push(1 + sector);
    }

    GeometryData { vertices, indices }
}

//表面
pub fn create_molecular_surface_geometry(
    protein: &Protein,
    grid_resolution: usize,
    probe_radius: f32,
) -> GeometryData {
    if protein.atoms.is_empty() {
        return GeometryData::new();
    }

    // 计算包围盒
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    for atom in &protein.atoms {
        let radius = atom.atom_type.radius() + probe_radius;
        min_x = min_x.min(atom.x - radius);
        max_x = max_x.max(atom.x + radius);
        min_y = min_y.min(atom.y - radius);
        max_y = max_y.max(atom.y + radius);
        min_z = min_z.min(atom.z - radius);
        max_z = max_z.max(atom.z + radius);
    }

    // 添加一些边界填充，防止表面不完整
    let padding = 2.0;
    min_x -= padding;
    max_x += padding;
    min_y -= padding;
    max_y += padding;
    min_z -= padding;
    max_z += padding;

    // 网格尺寸
    let grid_size_x = max_x - min_x;
    let grid_size_y = max_y - min_y;
    let grid_size_z = max_z - min_z;
    let nx = grid_resolution;
    let ny = grid_resolution;
    let nz = grid_resolution;

    // 密度数据
    let mut density_values = Vec::with_capacity(nx * ny * nz);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let x = min_x + (i as f32 / (nx - 1) as f32) * grid_size_x;
                let y = min_y + (j as f32 / (ny - 1) as f32) * grid_size_y;
                let z = min_z + (k as f32 / (nz - 1) as f32) * grid_size_z;

                // 计算该点的密度值
                let mut density = 0.0f32;
                for atom in &protein.atoms {
                    let dx = x - atom.x;
                    let dy = y - atom.y;
                    let dz = z - atom.z;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    let atom_radius = atom.atom_type.radius() + probe_radius;

                    // 使用平滑的密度函数
                    if distance <= atom_radius {
                        // 内部区域
                        density += 1.0;
                    } else if distance <= atom_radius * 1.5 {
                        // 过渡区域，使用平滑衰减
                        let normalized_distance = (distance - atom_radius) / (atom_radius * 0.5);
                        let falloff = 1.0 - normalized_distance;
                        density += falloff * falloff;
                    }
                }
                density_values.push(density);
            }
        }
    }

    // 使用 mcubes 生成网格
    let iso_level = 0.5f32; // 等值面阈值

    // 添加调试信息
    println!("网格参数: {}x{}x{}", nx, ny, nz);
    println!(
        "网格尺寸: {:.2} x {:.2} x {:.2}",
        grid_size_x, grid_size_y, grid_size_z
    );
    println!(
        "包围盒: ({:.2}, {:.2}, {:.2}) 到 ({:.2}, {:.2}, {:.2})",
        min_x, min_y, min_z, max_x, max_y, max_z
    );

    // 检查密度值范围
    let max_density = density_values.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_density = density_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    println!("密度范围: {:.2} 到 {:.2}", min_density, max_density);

    if max_density < iso_level {
        println!(
            "警告: 最大密度({:.2})小于等值面阈值({:.2})",
            max_density, iso_level
        );
        return GeometryData::new(); // 返回空几何体
    }

    let mc = MarchingCubes::new(
        (nx, ny, nz),                            // 网格点数量
        (grid_size_x, grid_size_y, grid_size_z), // 物理尺寸
        (nx as f32, ny as f32, nz as f32),       // 采样数量
        [min_x, min_y, min_z].into(),            // 偏移量
        density_values,                          // 密度数据
        iso_level,                               // 等值面值
    )
    .expect("Failed to create MarchingCubes");

    // 生成网格，使用合适的面方向
    let mesh = mc.generate(MeshSide::Both);

    // 转换为我们的几何数据格式
    let mut geometry = GeometryData::new();
    let surface_color = [0.6, 0.8, 1.0, 0.4]; // 浅蓝色，40%不透明度（更透明）

    // 转换顶点 - 正确的坐标缩放
    let mut min_vertex = [f32::INFINITY; 3];
    let mut max_vertex = [f32::NEG_INFINITY; 3];
    let mut min_grid = [f32::INFINITY; 3];
    let mut max_grid = [f32::NEG_INFINITY; 3];

    // 首先计算网格坐标的实际范围
    for vertex in &mesh.vertices {
        min_grid[0] = min_grid[0].min(vertex.posit.x);
        max_grid[0] = max_grid[0].max(vertex.posit.x);
        min_grid[1] = min_grid[1].min(vertex.posit.y);
        max_grid[1] = max_grid[1].max(vertex.posit.y);
        min_grid[2] = min_grid[2].min(vertex.posit.z);
        max_grid[2] = max_grid[2].max(vertex.posit.z);
    }

    // 现在将网格坐标转换为世界坐标
    for vertex in &mesh.vertices {
        // 归一化到0-1范围，然后映射到世界坐标
        let norm_x = (vertex.posit.x - min_grid[0]) / (max_grid[0] - min_grid[0]);
        let norm_y = (vertex.posit.y - min_grid[1]) / (max_grid[1] - min_grid[1]);
        let norm_z = (vertex.posit.z - min_grid[2]) / (max_grid[2] - min_grid[2]);

        let position = [
            min_x + norm_x * grid_size_x,
            min_y + norm_y * grid_size_y,
            min_z + norm_z * grid_size_z,
        ];

        // 跟踪坐标范围
        for i in 0..3 {
            min_vertex[i] = min_vertex[i].min(position[i]);
            max_vertex[i] = max_vertex[i].max(position[i]);
        }

        let normal = [vertex.normal.x, vertex.normal.y, vertex.normal.z];

        geometry.vertices.push(Vertex {
            position,
            normal,
            color: surface_color,
        });
    }

    // 转换索引
    geometry.indices = mesh.indices.iter().map(|&i| i as u32).collect();

    println!(
        "分子表面生成完成: {} 顶点, {} 三角形",
        geometry.vertices.len(),
        geometry.indices.len() / 3
    );

    // 输出顶点坐标范围
    if !geometry.vertices.is_empty() {
        println!(
            "网格坐标范围: ({:.2}, {:.2}, {:.2}) 到 ({:.2}, {:.2}, {:.2})",
            min_grid[0], min_grid[1], min_grid[2], max_grid[0], max_grid[1], max_grid[2]
        );
        println!(
            "表面顶点范围: ({:.2}, {:.2}, {:.2}) 到 ({:.2}, {:.2}, {:.2})",
            min_vertex[0],
            min_vertex[1],
            min_vertex[2],
            max_vertex[0],
            max_vertex[1],
            max_vertex[2]
        );
    }

    geometry
}

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 蛋白质渲染样式配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepresentationStyle {
    SpaceFilling,
    BallAndStick,
    MolecularSurface,
    Backbone,
    Cartoon,
}

/// 单个蛋白质的配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinConfig {
    pub path: PathBuf,
    pub styles: Vec<RepresentationStyle>,
    pub show_hydrogens: bool,
    pub color_scheme: Option<ColorScheme>,
    pub opacity: f32,
    pub visible: bool,
}

/// 颜色方案
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    /// 按元素类型着色（默认）
    Element,
    /// 按残基类型着色
    Residue,
    /// 按链着色
    Chain,
    /// 按二级结构着色
    SecondaryStructure,
    /// 自定义颜色
    Custom([f32; 3]),
}

/// 分子表面配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceConfig {
    /// 网格分辨率
    pub grid_resolution: usize,
    /// 探针半径
    pub probe_radius: f32,
    /// 等值面阈值
    pub iso_level: f32,
    /// 表面类型
    pub surface_type: SurfaceType,
}

/// 表面类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurfaceType {
    /// 溶剂可及表面
    SolventAccessible,
    /// van der Waals表面
    VanDerWaals,
    /// 分子表面
    Molecular,
}

/// 渲染配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    pub window_size: (u32, u32),
    pub window_title: String,
    pub background_color: [f32; 4],
    /// 是否启用抗锯齿
    pub enable_msaa: bool,
    /// 是否显示坐标轴
    pub show_axes: bool,
    /// 光照配置
    pub lighting: LightingConfig,
}

/// 光照配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingConfig {
    /// 环境光强度
    pub ambient_intensity: f32,
    /// 漫反射强度
    pub diffuse_intensity: f32,
    /// 镜面反射强度
    pub specular_intensity: f32,
    /// 光源方向
    pub light_direction: [f32; 3],
}

/// 相机配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
    pub move_speed: f32,
    pub rotation_speed: f32,
}

/// 主配置结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub proteins: Vec<ProteinConfig>,
    /// 表面配置
    pub surface: SurfaceConfig,
    pub render: RenderConfig,
    pub camera: CameraConfig,
}

impl Default for RepresentationStyle {
    fn default() -> Self {
        Self::BallAndStick
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self::Element
    }
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 32,
            probe_radius: 1.4,
            iso_level: 0.5,
            surface_type: SurfaceType::SolventAccessible,
        }
    }
}

impl Default for SurfaceType {
    fn default() -> Self {
        Self::SolventAccessible
    }
}

impl Default for LightingConfig {
    fn default() -> Self {
        Self {
            ambient_intensity: 0.3,
            diffuse_intensity: 0.7,
            specular_intensity: 0.5,
            light_direction: [-0.5, -1.0, -0.5],
        }
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            window_size: (1024, 768),
            window_title: "Protein Viewer".to_string(),
            background_color: [0.1, 0.1, 0.1, 1.0],
            enable_msaa: true,
            show_axes: true,
            lighting: LightingConfig::default(),
        }
    }
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 50.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_degrees: 45.0,
            near: 0.1,
            far: 1000.0,
            move_speed: 10.0,
            rotation_speed: 2.0,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            proteins: vec![],
            surface: SurfaceConfig::default(),
            render: RenderConfig::default(),
            camera: CameraConfig::default(),
        }
    }
}

impl AppConfig {
    /// 从文件加载配置
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;

        // 根据文件扩展名选择解析器
        let config = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::from_str(&content)?,
            Some("toml") => toml::from_str(&content)?,
            Some("yaml") | Some("yml") => serde_yaml::from_str(&content)?,
            _ => {
                // 默认尝试JSON
                serde_json::from_str(&content)?
            }
        };

        Ok(config)
    }

    /// 保存配置到文件
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::to_string_pretty(self)?,
            Some("toml") => toml::to_string_pretty(self)?,
            Some("yaml") | Some("yml") => serde_yaml::to_string(self)?,
            _ => serde_json::to_string_pretty(self)?,
        };

        std::fs::write(path, content)?;
        Ok(())
    }

    /// 创建默认配置文件的示例
    pub fn create_example_config() -> Self {
        Self {
            proteins: vec![
                ProteinConfig {
                    path: "protein.pdb".into(),
                    styles: vec![
                        RepresentationStyle::BallAndStick,
                        RepresentationStyle::MolecularSurface,
                    ],
                    show_hydrogens: false,
                    color_scheme: Some(ColorScheme::Element),
                    opacity: 1.0,
                    visible: true,
                },
                ProteinConfig {
                    path: "another_protein.pdb".into(),
                    styles: vec![RepresentationStyle::SpaceFilling],
                    show_hydrogens: false,
                    color_scheme: Some(ColorScheme::Chain),
                    opacity: 0.8,
                    visible: true,
                },
            ],
            surface: SurfaceConfig {
                grid_resolution: 64,
                probe_radius: 1.4,
                iso_level: 0.5,
                surface_type: SurfaceType::SolventAccessible,
            },
            render: RenderConfig {
                window_size: (1280, 720),
                window_title: "Multi-Protein Viewer".to_string(),
                background_color: [0.05, 0.05, 0.15, 1.0],
                enable_msaa: true,
                show_axes: true,
                lighting: LightingConfig {
                    ambient_intensity: 0.2,
                    diffuse_intensity: 0.8,
                    specular_intensity: 0.6,
                    light_direction: [-0.3, -1.0, -0.7],
                },
            },
            camera: CameraConfig {
                position: [0.0, 0.0, 80.0],
                target: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                fov_degrees: 50.0,
                near: 0.1,
                far: 2000.0,
                move_speed: 15.0,
                rotation_speed: 1.5,
            },
        }
    }
}

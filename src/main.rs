use learn_wgpu::{config::AppConfig, run_with_config};

fn main() -> anyhow::Result<()> {
    let config = match AppConfig::from_file(std::path::Path::new("config.json")) {
        Ok(config) => {
            println!("成功加载配置文件");
            config
        }
        Err(e) => {
            println!("加载配置文件失败: {}, 使用默认配置", e);

            // 默认配置
            let mut config = AppConfig::default();
            config.proteins.push(learn_wgpu::config::ProteinConfig {
                path: "protein.pdb".into(),
                styles: vec![learn_wgpu::config::RepresentationStyle::BallAndStick],
                show_hydrogens: false,
                color_scheme: Some(learn_wgpu::config::ColorScheme::Element),
                opacity: 1.0,
                visible: true,
            });

            // 保存示例配置文件
            if let Err(e) = config.save_to_file(std::path::Path::new("config_example.json")) {
                println!("保存示例配置文件失败: {}", e);
            } else {
                println!("已创建示例配置文件: config_example.json");
            }

            config
        }
    };

    run_with_config(config)
}

use cgmath::prelude::*;
use cgmath::{Matrix4, Point3, Rad, Vector3, perspective};
use std::f32::consts::FRAC_PI_2;
use winit::event::*;

//齐次坐标：（x,y,z,w）
//通过增加了一个w维度，使得平移也可通过矩阵乘法实现，简化计算

#[rustfmt::skip]
//把z轴（在NDC（Normalized Device Coordinates）立方体中）的范围从opengl标准的[-1,1]，映射到wgpu标准的[0,1]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

//Camera定义了一个视锥体，只有在视锥体内部的物体才会渲染到屏幕上
pub struct Camera {
    pub eye: Point3<f32>,    //相机的位置
    pub target: Point3<f32>, //相机看的目标
    pub up: Vector3<f32>,    //定义上方：y轴正半轴
    pub aspect: f32,         //渲染窗口的比例，防止图像变形
    pub fovy: f32,           //相机的视野角度
    pub znear: f32,          //近裁切面的距离，小于这个距离的物体将被裁切掉，不会渲染
    pub zfar: f32,           //远裁切面的距离，大于这个距离...不会渲染
}

impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>>(
        eye: V,
        target: V,
        up: Vector3<f32>,
        aspect: f32,
        fovy: Y,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            eye: eye.into(),
            target: target.into(),
            up,
            aspect,
            fovy: fovy.into().0,
            znear,
            zfar,
        }
    }

    pub fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        //view矩阵：将整个世界变换，以符合相机所看到的样子
        //proj矩阵：把所有在视锥体里的东西，变形塞进“标准立方体盒子”（NDC 空间）。
        //  1.GPU不好直接处理不规则的锥形空间，需要把空间标准化。
        //  2.使用透视投影的方式，在完成标准化的同时，也完美地创造了近大远小的视觉效果（近裁切面和立方体的面等大，远裁切面缩小到等大）。
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = perspective(Rad(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

pub struct CameraController {
    speed: f32,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_middle_pressed: bool,
    last_mouse_pos: (f64, f64),
    mouse_sensitivity: f32,
    distance: f32,
    theta: f32, // 水平角度
    phi: f32,   // 垂直角度
    target: Point3<f32>,
}

impl CameraController {
    pub fn new(speed: f32, mouse_sensitivity: f32) -> Self {
        Self {
            speed,
            is_left_pressed: false,
            is_right_pressed: false,
            is_middle_pressed: false,
            last_mouse_pos: (0.0, 0.0),
            mouse_sensitivity,
            distance: 50.0,
            theta: std::f32::consts::FRAC_PI_2,
            phi: 0.0,
            target: Point3::new(0.0, 0.0, 0.0),
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.is_left_pressed = *state == ElementState::Pressed;
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                self.is_right_pressed = *state == ElementState::Pressed;
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Middle,
                state,
                ..
            } => {
                self.is_middle_pressed = *state == ElementState::Pressed;
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let delta_x = position.x - self.last_mouse_pos.0;
                let delta_y = position.y - self.last_mouse_pos.1;

                if self.is_left_pressed {
                    // 旋转
                    self.theta -= (delta_x as f32) * self.mouse_sensitivity;
                    self.phi += (delta_y as f32) * self.mouse_sensitivity;

                    // 限制垂直角度
                    self.phi = self.phi.clamp(-FRAC_PI_2 + 0.1, FRAC_PI_2 - 0.1);
                }

                if self.is_right_pressed {
                    // 平移
                    let right = Vector3::new(self.theta.cos(), 0.0, -self.theta.sin());
                    let up = Vector3::new(0.0, 1.0, 0.0);

                    self.target += right * (delta_x as f32) * self.speed * 0.01;
                    self.target += up * -(delta_y as f32) * self.speed * 0.01;
                }

                self.last_mouse_pos = (position.x, position.y);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        self.distance -= y * self.speed;
                        self.distance = self.distance.max(1.0).min(1000.0);
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        self.distance -= pos.y as f32 * self.speed * 0.01;
                        self.distance = self.distance.max(1.0).min(1000.0);
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        // 球面坐标转换为笛卡尔坐标
        let x = self.distance * self.phi.cos() * self.theta.cos();
        let y = self.distance * self.phi.sin();
        let z = self.distance * self.phi.cos() * self.theta.sin();

        camera.eye = self.target + Vector3::new(x, y, z);
        camera.target = self.target;
    }

    pub fn set_target(&mut self, target: Point3<f32>) {
        self.target = target;
    }

    pub fn set_distance(&mut self, distance: f32) {
        self.distance = distance.max(1.0).min(1000.0);
    }

    pub fn reset(&mut self) {
        self.distance = 50.0;
        self.theta = std::f32::consts::FRAC_PI_2;
        self.phi = 0.0;
    }
}

//在更新camera之后得到新的view-projection矩阵，后续写入到GPU的Uniform Buffer（全局）中
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4], // 公开字段以便直接设置
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

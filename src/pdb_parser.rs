use nom::{IResult, bytes::complete::take, combinator::map};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{Request, RequestInit, RequestMode, Response};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[derive(Debug, Clone)]
pub struct Atom {
    pub serial: u32,
    pub name: String,
    pub residue_name: String,
    pub chain_id: char,
    pub residue_seq: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub element: String,
    pub atom_type: AtomType,
    pub molecule_type: MoleculeType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MoleculeType {
    Protein,
    DNA,
    RNA,
    Water,
    Ion,
    Other,
}

impl MoleculeType {
    pub fn from_residue_name(residue_name: &str) -> Self {
        match residue_name.trim().to_uppercase().as_str() {
            // 标准氨基酸
            "ALA" | "ARG" | "ASN" | "ASP" | "CYS" | "GLU" | "GLN" | "GLY" | "HIS" | "ILE" |
            "LEU" | "LYS" | "MET" | "PHE" | "PRO" | "SER" | "THR" | "TRP" | "TYR" | "VAL" |
            // 修饰氨基酸
            "MSE" | "SEC" | "PYL" => MoleculeType::Protein,
            
            // DNA碱基
            "DA" | "DT" | "DG" | "DC" | "DI" | "DU" => MoleculeType::DNA,
            
            // RNA碱基
            "A" | "U" | "G" | "C" | "I" => MoleculeType::RNA,
            
            // 水分子
            "HOH" | "WAT" | "H2O" => MoleculeType::Water,
            
            // 常见离子
            "NA" | "CL" | "K" | "MG" | "CA" | "ZN" | "FE" | "MN" | "CU" | "NI" => MoleculeType::Ion,
            
            _ => MoleculeType::Other,
        }
    }
    
    pub fn color(&self) -> [f32; 4] {
        match self {
            MoleculeType::Protein => [0.3, 0.7, 0.3, 1.0],  // 绿色 - 蛋白质
            MoleculeType::DNA => [0.3, 0.3, 0.7, 1.0],      // 蓝色 - DNA
            MoleculeType::RNA => [0.7, 0.3, 0.3, 1.0],      // 红色 - RNA
            MoleculeType::Water => [0.5, 0.8, 0.9, 0.3],    // 半透明蓝色 - 水
            MoleculeType::Ion => [0.9, 0.9, 0.3, 1.0],      // 黄色 - 离子
            MoleculeType::Other => [0.5, 0.5, 0.5, 1.0],    // 灰色 - 其他
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AtomType {
    Carbon,
    Nitrogen,
    Oxygen,
    Sulfur,
    Phosphorus,
    Hydrogen,
    Other(String),
}

impl AtomType {
    pub fn from_element(element: &str) -> Self {
        match element.trim().to_uppercase().as_str() {
            "C" => AtomType::Carbon,
            "N" => AtomType::Nitrogen,
            "O" => AtomType::Oxygen,
            "S" => AtomType::Sulfur,
            "P" => AtomType::Phosphorus,
            "H" => AtomType::Hydrogen,
            other => AtomType::Other(other.to_string()),
        }
    }

    pub fn color(&self) -> [f32; 4] {
        match self {
            AtomType::Carbon => [0.8, 0.8, 0.8, 1.0],     // 亮灰色，更容易看见
            AtomType::Nitrogen => [0.2, 0.2, 1.0, 1.0],   // 亮蓝色
            AtomType::Oxygen => [1.0, 0.2, 0.2, 1.0],     // 亮红色
            AtomType::Sulfur => [1.0, 1.0, 0.2, 1.0],     // 亮黄色
            AtomType::Phosphorus => [1.0, 0.6, 0.2, 1.0], // 亮橙色
            AtomType::Hydrogen => [0.9, 0.9, 0.9, 1.0],   // 浅灰色
            AtomType::Other(_) => [0.7, 0.7, 0.7, 1.0],   // 亮灰色
        }
    }

    pub fn radius(&self) -> f32 {
        match self {
            AtomType::Carbon => 1.7,
            AtomType::Nitrogen => 1.55,
            AtomType::Oxygen => 1.52,
            AtomType::Sulfur => 1.8,
            AtomType::Phosphorus => 1.8,
            AtomType::Hydrogen => 1.2,
            AtomType::Other(_) => 1.5,
        }
    }
}

#[derive(Debug)]
pub struct Bond {
    pub atom1_index: usize,
    pub atom2_index: usize,
    pub bond_type: BondType,
}

#[derive(Debug, Clone)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

#[derive(Debug)]
pub struct Protein {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
}

impl Protein {
    pub fn new() -> Self {
        Self {
            atoms: Vec::new(),
            bonds: Vec::new(),
        }
    }

    pub fn add_atom(&mut self, atom: Atom) {
        self.atoms.push(atom);
    }

    pub fn generate_bonds(&mut self) {
        // 简单的化学键生成算法
        let max_bond_distance = 2.0; // 最大键长度

        for i in 0..self.atoms.len() {
            for j in i + 1..self.atoms.len() {
                let atom1 = &self.atoms[i];
                let atom2 = &self.atoms[j];

                let distance = ((atom1.x - atom2.x).powi(2)
                    + (atom1.y - atom2.y).powi(2)
                    + (atom1.z - atom2.z).powi(2))
                .sqrt();

                if distance < max_bond_distance {
                    // 排除氢原子之间的键
                    if atom1.atom_type != AtomType::Hydrogen
                        || atom2.atom_type != AtomType::Hydrogen
                    {
                        self.bonds.push(Bond {
                            atom1_index: i,
                            atom2_index: j,
                            bond_type: BondType::Single,
                        });
                    }
                }
            }
        }
    }

    pub fn center(&self) -> [f32; 3] {
        if self.atoms.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        let sum = self.atoms.iter().fold([0.0, 0.0, 0.0], |acc, atom| {
            [acc[0] + atom.x, acc[1] + atom.y, acc[2] + atom.z]
        });

        let count = self.atoms.len() as f32;
        [sum[0] / count, sum[1] / count, sum[2] / count]
    }

    /// 计算蛋白质的包围盒半径（从中心到最远原子的距离）
    pub fn bounding_radius(&self) -> f32 {
        let center = self.center();
        let mut max_distance = 0.0;

        for atom in &self.atoms {
            let dx = atom.x - center[0];
            let dy = atom.y - center[1];
            let dz = atom.z - center[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            if distance > max_distance {
                max_distance = distance;
            }
        }

        max_distance
    }
}

pub fn parse_pdb_line(line: &str) -> IResult<&str, Option<Atom>> {
    if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
        return Ok((line, None));
    }

    let (input, _) = take(6usize)(line)?;
    let (input, serial) = map(take(5usize), |s: &str| s.trim().parse::<u32>().unwrap_or(0))(input)?;
    let (input, _) = take(1usize)(input)?;
    let (input, name) = map(take(4usize), |s: &str| s.trim().to_string())(input)?;
    let (input, _) = take(1usize)(input)?;
    let (input, residue_name) = map(take(3usize), |s: &str| s.trim().to_string())(input)?;
    let (input, _) = take(1usize)(input)?;
    let (input, chain_id) = map(take(1usize), |s: &str| s.chars().next().unwrap_or(' '))(input)?;
    let (input, residue_seq) =
        map(take(4usize), |s: &str| s.trim().parse::<u32>().unwrap_or(0))(input)?;
    let (input, _) = take(4usize)(input)?;
    let (input, x) = map(take(8usize), |s: &str| {
        s.trim().parse::<f32>().unwrap_or(0.0)
    })(input)?;
    let (input, y) = map(take(8usize), |s: &str| {
        s.trim().parse::<f32>().unwrap_or(0.0)
    })(input)?;
    let (input, z) = map(take(8usize), |s: &str| {
        s.trim().parse::<f32>().unwrap_or(0.0)
    })(input)?;

    // 跳过占用率和温度因子
    let (input, _) = take(12usize)(input)?;

    // 元素符号
    let element = if input.len() >= 2 {
        let elem = input[0..2].trim().to_string();
        if elem.is_empty() {
            // 如果从PDB位置读取的元素为空，从原子名称推断
            name.chars()
                .filter(|c| c.is_alphabetic())
                .take(1) // 只取第一个字母作为元素符号
                .collect::<String>()
        } else {
            elem
        }
    } else {
        // 从原子名称推断元素
        name.chars()
            .filter(|c| c.is_alphabetic())
            .take(1) // 只取第一个字母作为元素符号
            .collect::<String>()
    };

    let atom = Atom {
        serial,
        name,
        residue_name: residue_name.clone(),
        chain_id,
        residue_seq,
        x,
        y,
        z,
        element: element.clone(),
        atom_type: AtomType::from_element(&element),
        molecule_type: MoleculeType::from_residue_name(&residue_name),
    };

    Ok((input, Some(atom)))
}

pub fn parse_pdb(content: &str) -> anyhow::Result<Protein> {
    let mut protein = Protein::new();

    for line in content.lines() {
        if let Ok((_, Some(atom))) = parse_pdb_line(line) {
            protein.add_atom(atom);
        }
    }

    // 生成化学键
    protein.generate_bonds();

    Ok(protein)
}

pub fn load_pdb_file(path: &str) -> anyhow::Result<Protein> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let content = std::fs::read_to_string(path)?;
        parse_pdb(&content)
    }
    
    #[cfg(target_arch = "wasm32")]
    {
        anyhow::bail!("Use load_pdb_from_url or load_pdb_from_blob instead in WASM")
    }
}

/// 从URL加载PDB文件 (WASM版本)
#[cfg(target_arch = "wasm32")]
pub async fn load_pdb_from_url(url: &str) -> Result<Protein, wasm_bindgen::JsValue> {
    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_str(url)).await?;
    let resp: Response = resp_value.dyn_into().unwrap();
    
    if !resp.ok() {
        return Err(wasm_bindgen::JsValue::from_str(&format!("HTTP error: {}", resp.status())));
    }
    
    let text = JsFuture::from(resp.text().unwrap()).await?;
    let content = text.as_string().unwrap();
    
    parse_pdb(&content).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

/// 从Blob加载PDB文件 (WASM版本)  
#[cfg(target_arch = "wasm32")]
pub async fn load_pdb_from_blob(blob: &web_sys::Blob) -> Result<Protein, wasm_bindgen::JsValue> {
    let text_promise = blob.text();
    let text_value = JsFuture::from(text_promise).await?;
    let content = text_value.as_string().unwrap();
    
    parse_pdb(&content).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

/// 从File对象加载PDB文件 (WASM版本)
#[cfg(target_arch = "wasm32")]
pub async fn load_pdb_from_file(file: &web_sys::File) -> Result<Protein, wasm_bindgen::JsValue> {
    let text_promise = file.text();
    let text_value = JsFuture::from(text_promise).await?;
    let content = text_value.as_string().unwrap();
    
    parse_pdb(&content).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom_line() {
        let line =
            "ATOM      1  N   ALA A   1      20.154  16.967  14.720  1.00 15.02           N  ";
        let (_, atom) = parse_pdb_line(line).unwrap();

        assert!(atom.is_some());
        let atom = atom.unwrap();
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.name, "N");
        assert_eq!(atom.residue_name, "ALA");
        assert_eq!(atom.chain_id, 'A');
        assert_eq!(atom.residue_seq, 1);
        assert!((atom.x - 20.154).abs() < 0.001);
        assert!((atom.y - 16.967).abs() < 0.001);
        assert!((atom.z - 14.720).abs() < 0.001);
    }
}

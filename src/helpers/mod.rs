use crate::{remove_file, File, Path, Write};

/// Returns Ok(()) if file is saved Err(String) otherwise
///
/// # Argument
///
/// * data - vectorized data to save
/// * f_path - path where file should be saved
///
pub fn save_file(data: &Vec<u8>, f_path: &String) -> Result<(), String> {
    let path = Path::new(f_path);
    if let Ok(mut _f) = File::create(path) {
        if let Ok(_) = _f.write(&data) {
            info!("File in path {:?} has been created and content of length {:?} bytes has been written", &path, &data.len());
            return Ok(());
        }
    };
    Err(format!("File cannot be saved."))
}

/// Returns Ok(()) if file deleted Err(String) otherwise
///
/// # Arguments
///
/// * f_path - path to file to delete
///
pub fn delete_file(f_path: &String) -> Result<(), String> {
    let path = Path::new(f_path);
    if let Ok(_) = remove_file(&path) {
        info!("File in path {:?} has been deleted", &path);
        return Ok(());
    };
    Err(format!("File cannot be deleted."))
}

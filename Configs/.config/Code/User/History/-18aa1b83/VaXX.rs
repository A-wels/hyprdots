use std::fs::OpenOptions;
use std::io::{Error, Read, Write};

pub fn save_timestamp_linux(timestamp: u32) -> Result<(), Error> {
    // create ~/.CrossGPodder if it doesn't exist
    let home_dir = dirs::home_dir();
    let gpodder_dir = format!("{}/.CrossGPodder", home_dir.unwrap().display());
    let _ = std::fs::create_dir_all(&gpodder_dir);

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(format!("{}/timestamp.txt", &gpodder_dir))?;
    println!("Writing to: {:?}", file);
    file.write_all(timestamp.to_string().as_bytes())?;
    Ok(())
}

pub fn load_timestamp_linux() -> Result<u32, Error> {
    // create ~/.CrossGPodder if it doesn't exist
    let home_dir: Option<std::path::PathBuf> = dirs::home_dir();
    let gpodder_dir = format!("{}/.CrossGPodder", home_dir.unwrap().display());
    let _ = std::fs::create_dir_all(&gpodder_dir);

    let file = OpenOptions::new()
        .read(true)
        .create(true)
        .open("/home/alex/.CrossGPodder/timestamp.txt");

    let mut contents = String::new();
    println!("{:?}", file);
    return Ok(0);
}
use std::fs::OpenOptions;
use std::io::{Error, Read, Write};

pub fn save_timestamp_linux(timestamp: u32) -> Result<(), Error> {

    // create ~/.CrossGPodder if it doesn't exist
    let home_dir = dirs::home_dir();
    let gpodder_dir = format!("{}/.CrossGPodder", home_dir.unwrap().display());
    let _ = std::fs::create_dir_all(&gpodder_dir);
    println!("Writing timestamp to file...");
    println!("{}", &timestamp);

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(format!("{}/timestamp.txt", &gpodder_dir))?;
    file.write_all(timestamp.to_string().as_bytes())?;
    Ok(())
}

pub fn load_timestamp_linux() -> u32 {
        // create ~/.CrossGPodder if it doesn't exist
        let home_dir = dirs::home_dir();
        let gpodder_dir = format!("{}/.CrossGPodder", home_dir.unwrap().display());
        let _ = std::fs::create_dir_all(&gpodder_dir);
    let mut file = OpenOptions::new()
        .read(true)
        .create(true)
        .open(format!("{}/timestamp.txt", &gpodder_dir));
    let mut contents = String::new();
    file.read_to_string(&mut contents);
    let timestamp = contents.parse::<u32>().unwrap();
    Ok(timestamp)
}
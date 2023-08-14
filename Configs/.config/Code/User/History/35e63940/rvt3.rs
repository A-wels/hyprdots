mod structs;
use crate::structs::server::Server;

fn main() {
    
    // Define the server
    let server = Server {
        url: "cloud.a-wels.de".to_string(),
        protocol: "https".to_string(),
    };

    print!("Sending GET Request to the server...");
    // GET Request to the server to get subscriptions
    let _subscriptions = server.get_subscriptions();
    println!("Done!");
   loop {
       
   }
}

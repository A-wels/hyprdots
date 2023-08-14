mod structs;
use crate::structs::server::Server;

#[tokio::main]
async fn main() {
    
    // Define the server
    let server = Server::new(
         "cloud.a-wels.de".to_string(),
            "https".to_string(),
            "alex".to_string(),
            "Eaydmhf9qxazfhbQRSc7p9wdQUgfrhyqYJ22VAmrw2QWZqm".to_string(),

    );

    // If no token is available, login to the server
    if server.token == "" {
        println!("No token available. Logging in...");
        server.start_login();
    }

    print!("Sending GET Request to the server...");
    // GET Request to the server to get subscriptions
    let _subscriptions =  server.get_subscriptions().await;
    println!("Done!");

}

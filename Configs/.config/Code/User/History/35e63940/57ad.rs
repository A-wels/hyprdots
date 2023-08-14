mod structs;
use crate::structs::server::Server;
use crate::structs::token::TokenStorage;

#[tokio::main]
async fn main() {
    
    // Define the server
    let mut server = Server::new(
         "cloud.a-wels.de".to_string(),
            "https".to_string(),
            "alex".to_string(),

    );
    server.token = TokenStorage::get_current_token_linux().await.unwrap();

    // If no token is available, login to the server
    if server.token.token == "" {
        println!("No token available. Logging in...");
       // server.start_login().await;
        // Store the token in the keyring
        //let _ = TokenStorage::store_current_token_linux(server.token).await;
        // Load the token from the keyring
        println!("Token: {:?}", server.token);


    }

    print!("Sending GET Request to the server...");
    // GET Request to the server to get subscriptions
    let _subscriptions: Result<(), reqwest::Error> =  server.get_subscriptions().await;
    println!("Done!");

}

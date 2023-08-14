use reqwest::Error;
use webbrowser;

use super::{Subscription, Token};

// Class containing server information: URL, port, etc.
#[derive(Debug, Clone)]
pub struct Server {
    pub url: String,
    pub protocol: String,
    pub username: String,
    pub client: reqwest::Client,
    pub last_sync: u32,
    pub subscriptions: Vec<Subscription>,
    // List of subscriptions

    // Authentication token, default is empty
    pub token: Token,
}

impl Server {
    // initialize the server struct
    pub fn new(url: String, protocol: String, username: String) -> Server {
        // Create a new client
        let client = reqwest::Client::new();
        // return the server struct
        Server {
            url,
            protocol,
            username,
            client,
            token: Token { token: "".to_string(), expires: "".to_string() },
            last_sync: 0,
            subscriptions: Vec::new(),
        }
    }

    pub async fn start_login(&mut self) -> Result<(), Error> {
        // Send POST request to Nextcloud server
        let url = format!("{}://{}/index.php/login/v2", self.protocol, self.url);
        // fetch the response and grab the login field
        let response = self
            .client
            .post(url)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let login = response.split("login\":\"").collect::<Vec<&str>>()[1]
            .split("\"")
            .collect::<Vec<&str>>()[0];
        let token = response.split("token\":\"").collect::<Vec<&str>>()[1]
            .split("\"")
            .collect::<Vec<&str>>()[0];
        let poll_url = response.split("endpoint\":\"").collect::<Vec<&str>>()[1]
            .split("\"")
            .collect::<Vec<&str>>()[0];

        // Open the login page in the browser
        if webbrowser::open(login).is_ok() {
            // Poll the endpoint every second to get auth token

            let poll_url = poll_url.to_string();
            let token = token.to_string();
            let request_body = [("token", token)];

            loop {
                let response = self
                    .client
                    .post(&poll_url)
                    .form(&request_body)
                    .send()
                    .await?;

                if response.status().is_success() {
                    println!("Success!");
                    let response_text = response.text().await?;

                    // Get the fields server, loginName and appPassword. Will not be available if authorization is not completed yet.
                    // In this case, dont panic, just wait for 1 second and try again
                    // Assuming `response` is of type `&str`

                    let login_name =
                        if let Some(value) = response_text.split("loginName\":\"").nth(1) {
                            value.split("\"").next().unwrap_or_default()
                        } else {
                            ""
                        };

                    let app_password =
                        if let Some(value) = response_text.split("appPassword\":\"").nth(1) {
                            value.split("\"").next().unwrap_or_default()
                        } else {
                            ""
                        };

                    if login_name != "" && app_password != "" {
                        // Set the username and password
                        self.username = login_name.to_string();
                        let token = Token{
                            token: app_password.to_string(),
                            expires: "".to_string(),
                        };
                        self.token = token;
                        break;
                    }
                }

                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        } else {
            println!("Failed to open login page in browser!");
        }

        Ok(())
        //
    }

    // function to get subscriptions from the server
    pub async fn get_subscriptions(&mut self) -> Result<(), Error> {
        // Login to the server

        println!("Sending GET Request to the server...");
        // GET Request to the url
        let get_url = format!(
            "{}://{}/index.php/apps/gpoddersync/subscriptions",
            self.protocol, self.url
        );
        println!("{}", get_url);
        let response = self
            .client
            .get(get_url)
            .basic_auth(&self.username, Some(&self.token.token))
            .send()
            .await?
            .text()
            .await?;
        println!("Response: {}", response);
        // Extract the timestamp from the response
        let timestamp = response.split("\"timestamp\":").collect::<Vec<&str>>()[1]
            .split("}")
            .collect::<Vec<&str>>()[0];

        self.last_sync = timestamp.parse::<u32>().unwrap();

        println!("Last sync: {}", self.last_sync);

        // Extract the new subscriptions from the response
        let subscriptions_string = response.split("add\":[").collect::<Vec<&str>>()[1]
            .split("]")
            .collect::<Vec<&str>>()[0];

        // Split the subscriptions into a vector
        let subscriptions = subscriptions_string.split(",").collect::<Vec<&str>>();
        // Iterate over the subscriptions
        for s in subscriptions {
            self.subscriptions.push(Subscription { url: s.to_string() })
        }
        println!("Subscriptions: {:?}", self.subscriptions);

        // Remove removed subscriptions
        let removed_subscriptions_string = response.split("remove\":[").collect::<Vec<&str>>()[1]
            .split("]")
            .collect::<Vec<&str>>()[0];

        // Split the subscriptions into a vector
        let removed_subscriptions = removed_subscriptions_string
            .split(",")
            .collect::<Vec<&str>>();
        // Iterate over the subscriptions, save the index of the subscription to remove
        let mut indexes_to_remove: Vec<usize> = Vec::new();
        for s in removed_subscriptions {
            let mut index = 0;
            for subscription in &self.subscriptions {
                if subscription.url == s.to_string() {
                    indexes_to_remove.push(index);
                }
                index += 1;
            }
        }
        // Remove the subscriptions backwards, so the indexes don't change
        indexes_to_remove.sort();
        indexes_to_remove.reverse();
        for i in indexes_to_remove {
            self.subscriptions.remove(i);
        }
        Ok(())
    }
}

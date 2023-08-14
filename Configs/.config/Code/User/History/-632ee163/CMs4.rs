use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Token {
    pub token: String,
    pub expires: String,
}

pub struct TokenStorage {}
impl TokenStorage {
    // load the token from keyring and return it
    pub async fn get_current_token_linux() -> oo7::Result<Token> {
        let keyring = oo7::Keyring::new().await?;

        // find the token in the keyring
        let items = keyring
            .search_items(HashMap::from([("attribute", "token_attribute")]))
            .await?;

        println!("{:?}", items);

        // check if items.first() is not None
        if let Some(item) = items.first() {
            // get the token from the item
            let token = std::str::from_utf8(item.secret().await?.to_owned().as_slice()).unwrap();
            println!("{:?}", token);
            // deserialize the token
            //  let token: Token = serde_json::from_str(&token).unwrap();
            // return the token
            return Ok(Token {
                token: token.to_string(),
                expires: " ()".to_string(),
            });
        }
        println!("-----------------------------------");

        return Ok(Token {
            token: "()".to_string(),
            expires: " ()".to_string(),
        });
    }

    pub async fn store_current_token_linux(token: Token) -> oo7::Result<()> {
        let keyring = oo7::Keyring::new().await?;
        let serialized_token = serde_json::to_string(&token).unwrap();

        keyring
            .create_item(
                "Token",
                HashMap::from([("attribute", "token_attribute")]),
                serialized_token,
                true,
            )
            .await?;

        Ok(())
    }
}

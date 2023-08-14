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
        let item = match keyring
            .search_items(HashMap::from([("attribute", "token_attribute")]))
            .await?
            .first()
        {
            Some(item) => item,
            None => return Err(oo7::Error::Generic("Token not found".to_string())),
        };

        let stored_token_str = std::str::from_utf8(item.secret().await?.as_slice())
            .map_err(|_| Err("Token not found"));

     
        println!("{:?}", token);
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

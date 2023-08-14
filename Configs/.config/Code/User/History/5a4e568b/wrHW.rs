#[derive(Debug, Clone)]
pub struct Subscription {
    pub url: String,
}

impl Subscription {
    pub fn new(url: String) -> Subscription {
        Subscription { url }
    }
}
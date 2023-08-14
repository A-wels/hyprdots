use std::{sync::{Arc, Mutex}, collections::HashMap};

use axum::{
    routing::get,
    Router, Extension,AddExtensionLayer
};

type Scoreboard = Arc<Mutex<HashMap<i32, i32>>>;

// update the score for a team
async fn update_score(Path(team): Path<i32>, Extension(scoreboard): Extension<Scoreboard>) -> &'static str {
    let mut scores = scoreboard.lock().unwrap();
    let score = scores.entry(team).or_insert(0);
    *score += 1;
    "Score updated successfully!"
}
ore(Path(team): Path<i32>, Extension(scoreboard))

#[tokio::main]
async fn main() {
    // Create a scoreboard and wrap it in an Arc and Mutex to make it shareable between threads
    let scoreboard: Scoreboard = Arc::new(Mutex::new(HashMap::new()));

    // Build the Axum app
    let app = route("/", get(handler));
    let app = app
        .route("/1", get(update_score))
        .route("/2", get(update_score))
        .route("/3", get(update_score))
        .route("/4", get(update_score))
        .layer(AddExtensionLayer::new(scoreboard));

    // Run the Axum server
    let addr = "0.0.0.0:3001".parse().unwrap();
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
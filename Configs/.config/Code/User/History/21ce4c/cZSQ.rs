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
    // build our application with a single route
    let app = Router::new().route("/", get(|| async { "Hello, World!" }));

    // run it with hyper on localhost:3000
    axum::Server::bind(&"0.0.0.0:3001".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
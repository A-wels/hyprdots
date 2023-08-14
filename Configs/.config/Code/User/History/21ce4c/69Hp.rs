use std::collections::HashMap;

use axum::{
    routing::get,
    Router,
};


#[tokio::main]
async fn main() {
struct scoreboard {
    scoreTeam1: i32,
    scoreTeam2: i32,
    scoreTeam3: i32,
    scoreTeam4: i32,
}

    // build our application with a single route
    let app = Router::new().route("/", get(|| async { "Hello, World!" }));

    // run it with hyper on localhost:3000
    axum::Server::bind(&"0.0.0.0:3001".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
use std::collections::HashMap;

use axum::{
    routing::get,
    Router,
};
struct scoreboard {
    scoreTeam1: i32,
    scoreTeam2: i32,
    scoreTeam3: i32,
    scoreTeam4: i32,
}

#[tokio::main]
async fn main() {
    let mut scoreboard = scoreboard {
        scoreTeam1: 0,
        scoreTeam2: 0,
        scoreTeam3: 0,
        scoreTeam4: 0,
    };

    // build our application with a single route
    let app = Router::new().route("/", get(|| async { "Hello, World!" }));

    // run it with hyper on localhost:3000
    axum::Server::bind(&"0.0.0.0:3001".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
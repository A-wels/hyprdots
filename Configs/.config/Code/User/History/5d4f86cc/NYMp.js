// when an element of the class item is clicked, the function is called.
// get the id of the element and pass it to the function
// the function will then increase the points of the team by 1

document.addEventListener('click', function (event) {
    console.log(event.target.parent.classList)
    if (event.target.classList.contains('item')) {
        increasePoints(event.target.id);
    }
}
);
// get child with class punkte and increase the innerHTML by 1
function increasePoints(id) {
    let team = document.getElementById(id);
    console.log(id)
    let points = team.getElementsByClassName('punkte')[0];
    points.innerHTML = parseInt(points.innerHTML) + 1;
}


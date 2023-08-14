// when an element of the class item is clicked, the function is called.
// get the id of the element and pass it to the function
// the function will then increase the points of the team by 1

document.addEventListener('click', function (event) {
    if (event.target.parentElement.classList.contains('item')) {

        // switch:  if the id of the element is team1, then increase the points of team1
        // if the id of the element is team2, then increase the points of team2
        switch (event.target.parentElement.id) {
            case 'item1':
                increasePoints('punkte1');
                break;
            case 'item2':
                increasePoints('punkte2');
                break;
            case 'item3':
                increasePoints('punkte3');
                break;
            case 'item4':
                increasePoints('punkte4');
                break;
        }
    }
});

function increasePoints(team) {
    // get the points of the team
    let points = document.getElementById(team).innerHTML;

    // increase the points by 1
    points++;

    // update the points of the team
    document.getElementById(team).innerHTML = points;
}

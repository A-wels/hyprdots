// when an element of the class item is clicked, the function is called.
// get the id of the element and pass it to the function
// the function will then increase the points of the team by 1

document.addEventListener('click', function (event) {
    if (event.target.classList.contains('item')) {
        increasePoints(event.target.id);
    }
}
);
// get child with class punkte and increase the innerHTML by 1

const currentYear = document.querySelector('#year');
if (currentYear) {
  currentYear.textContent = new Date().getFullYear();
}

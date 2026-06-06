const currentYear = document.querySelector('#year');
if (currentYear) {
  currentYear.textContent = new Date().getFullYear();
}

var eh = document.getElementById('email-holder');
if (eh) {
  var addr = ['pppadiac', 'gmail.com'].join('@');
  var a = document.createElement('a');
  a.href = 'mail' + 'to:' + addr;
  a.textContent = addr;
  eh.appendChild(a);
}

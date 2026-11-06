const ATTR_OPEN = 'nav-open';
const $trigger = document.getElementById('nav-trigger');
const $mask = document.getElementById('mask');

function setOpen(open) {
  document.body.toggleAttribute(ATTR_OPEN, open);
  $mask.classList.toggle('d-none', !open);
}

export function initNav() {
  if (!$trigger) {
    return;
  }

  $trigger.onclick = () => setOpen(!document.body.hasAttribute(ATTR_OPEN));
  $mask.onclick = () => setOpen(false);
}

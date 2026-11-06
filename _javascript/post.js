import { basic, initTopbar, initNav } from './modules/layouts';

import {
  loadImg,
  imgPopup,
  initLocaleDatetime,
  initClipboard,
  initToc,
  loadMermaid
} from './modules/components';

loadImg();
initToc();
imgPopup();
initNav();
initLocaleDatetime();
initClipboard();
initTopbar();
loadMermaid();
basic();

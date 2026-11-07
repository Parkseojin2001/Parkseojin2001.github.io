import { basic, initNav, initTopbar } from './modules/layouts';
import {
  loadImg,
  imgPopup,
  initClipboard,
  loadMermaid
} from './modules/components';

loadImg();
imgPopup();
initNav();
initTopbar();
initClipboard();
loadMermaid();
basic();

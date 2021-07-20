/**
 * Custom JS for https://iwasakishuto.github.io/PyVideoEditor
 *
 * Copyright : MIT, 2021 iwasakishuto
 * Twitter : https://twitter.com/cabernet_rock
 * Github : https://github.com/iwasakishuto
 *
*/

function isFunction(func) {
  return func && {}.toString.call(func) === '[object Function]';
 }
// Update the "window.onload"
function addOnLoad(fn){
  if (isFunction(fn)){
    var old = window.onload;
    window.onload = isFunction(old)
      ? function(){old();fn();}
      : fn
  }
}

(function (){
  $(function(){
    var items = document.querySelectorAll('div.toctree-wrapper.compound li[class*="toctree-"]');
    items.forEach((item) => {
      var text  = item.querySelector("a").innerHTML
      var text_components = text.split(".");
      var num_components = text_components.length;
      if (num_components>0){
        text = text_components[num_components-1];
      }
      text = text.replace(/(.*)\spackage/g,' <span class="package-name">$1 package</span>')
                .replace(/(.*)\smodule/g, '<span class="program-name">$1.py</span>')
                .replace(/(Subpackages|Submodules)/g,'<span class="package-subtitle">$1</span>')
                .replace(/(Module\scontents)/g, '<span class="module-contents">$1</span>');
      a = item.querySelector("a")
      a.innerHTML = text
      if (!text.includes("<span")) a.classList.add("chapter")
    });
  });
})(jQuery);

function hide_typing_info(){
  sig_params = document.querySelectorAll("em.sig-param > span")
  sig_params.forEach(function(e,i){
      if (e.textContent==":"){
          e.remove()
          typing = sig_params[i+1].textContent
          sig_params[i+1].remove()
          sig_params[i-1].setAttribute("aria-label", typing)
      }
  })
}

addOnLoad(hide_typing_info)
// Click-to-coordinates image component.
// Based on streamlit-image-coordinates 0.2.2 (MIT, Zachary Blackwood), with the
// resize handler replaced instead of accumulated: the original added a new
// window "resize" listener on every render, so handlers from a previous image
// kept firing and stretched the current image to the old image's size.

let activeResize = null;

function sendValue(value) {
  Streamlit.setComponentValue(value)
}

function clickListener(event) {
  const img = document.getElementById("image");
  const rect = img.getBoundingClientRect();
  sendValue({
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
    width: rect.width,
    height: rect.height,
    unix_time: Date.now()
  });
}

function mouseDownListener(downEvent) {
  const img = document.getElementById("image");
  const downRect = img.getBoundingClientRect();
  const x1 = downEvent.clientX - downRect.left;
  const y1 = downEvent.clientY - downRect.top;

  window.addEventListener("mouseup", (upEvent) => {
    const rect = img.getBoundingClientRect();
    sendValue({
      x1: x1, y1: y1,
      x2: upEvent.clientX - rect.left, y2: upEvent.clientY - rect.top,
      width: rect.width, height: rect.height,
      unix_time: Date.now()
    });
  }, {once: true});
}

function onRender(event) {
  const {src, height, width, use_column_width, click_and_drag} = event.detail.args;
  const img = document.getElementById("image");

  function resizeImage() {
    if (!img.naturalWidth) {
      return;
    }
    img.classList.remove("auto", "fullWidth");
    img.removeAttribute("width");
    img.removeAttribute("height");

    if (use_column_width === "always" || use_column_width === true) {
      img.classList.add("fullWidth");
    } else if (use_column_width === "auto") {
      img.classList.add("auto");
    } else {
      let w = width;
      let h = height;
      if (!w && !h) {
        w = img.naturalWidth;
        h = img.naturalHeight;
      } else if (!h) {
        h = w * img.naturalHeight / img.naturalWidth;
      } else if (!w) {
        w = h * img.naturalWidth / img.naturalHeight;
      }
      img.width = Math.round(w);
      img.height = Math.round(h);
    }

    Streamlit.setFrameHeight(img.getBoundingClientRect().height);
  }

  if (activeResize) {
    window.removeEventListener("resize", activeResize);
  }
  activeResize = resizeImage;
  window.addEventListener("resize", activeResize);

  img.onload = resizeImage;
  if (img.src !== src) {
    img.src = src;
  } else if (img.complete) {
    resizeImage();
  }

  if (click_and_drag) {
    img.onclick = null;
    img.onmousedown = mouseDownListener;
  } else {
    img.onmousedown = null;
    img.onclick = clickListener;
  }
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()

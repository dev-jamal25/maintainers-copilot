/* Maintainer's Copilot widget loader.
 *
 * Usage on a host page (one tag):
 *   <script src="https://api.example/widget.js" data-widget-id="w_abc123"></script>
 *
 * Reads data-widget-id, fetches the public widget config from the API, and (if active) injects an
 * iframe pointing at the API embed route. Fails safe + visibly: a blocked origin makes the config
 * fetch fail CORS (the API only sets Access-Control-Allow-Origin for allowed origins), which lands in
 * the catch and logs without breaking the host page. Exposes no secrets.
 */
(function () {
  "use strict";
  var script = document.currentScript;
  if (!script) {
    return;
  }
  var widgetId = script.getAttribute("data-widget-id");
  var apiBase = script.getAttribute("data-api-base") || new URL(script.src).origin;
  var TAG = "[maintainers-copilot]";

  if (!widgetId) {
    console.error(TAG + " missing data-widget-id; widget not loaded");
    return;
  }

  var configUrl = apiBase + "/widgets/" + encodeURIComponent(widgetId) + "/config";
  fetch(configUrl)
    .then(function (response) {
      if (!response.ok) {
        throw new Error("config request failed: " + response.status);
      }
      return response.json();
    })
    .then(function (config) {
      if (!config.is_active) {
        console.warn(TAG + " widget is inactive; not rendering");
        return;
      }
      var iframe = document.createElement("iframe");
      iframe.src = apiBase + "/widgets/" + encodeURIComponent(widgetId) + "/embed";
      iframe.title = "Maintainer's Copilot";
      iframe.setAttribute("aria-label", "Maintainer's Copilot chat");
      iframe.style.position = "fixed";
      iframe.style.zIndex = "2147483000";
      iframe.style.border = "0";
      iframe.style.width = "380px";
      iframe.style.height = "560px";
      iframe.style.maxWidth = "calc(100vw - 32px)";
      iframe.style.bottom = "20px";
      var position = config.position || "bottom-right";
      if (position.indexOf("left") >= 0) {
        iframe.style.left = "20px";
      } else {
        iframe.style.right = "20px";
      }
      document.body.appendChild(iframe);

      // Resize messages are only trusted from the API origin (the iframe's origin).
      window.addEventListener("message", function (event) {
        if (event.origin !== apiBase) {
          return;
        }
        var data = event.data || {};
        if (data.type === "maintainers-copilot:resize" && typeof data.height === "number") {
          iframe.style.height = Math.min(data.height, window.innerHeight - 40) + "px";
        }
      });
    })
    .catch(function (error) {
      console.error(
        TAG + " could not load widget (origin not allowed or config missing): " + error.message
      );
    });
})();

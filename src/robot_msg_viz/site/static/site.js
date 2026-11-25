/* robot_msg_viz gallery: theme, drawer, tabs, the All/Supported filter, search, copy buttons and the viewer player.
   No dependencies; every storage access is wrapped, so a private window or blocked storage just forgets the pick. */
(function () {
  var root = document.documentElement;
  var body = document.body;
  var base = body.dataset.root || "";

  function store(key, value) { try { localStorage.setItem(key, value); } catch (e) {} }

  // theme
  var themeBtn = document.querySelector(".theme-btn");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      store("theme", next);
    });
  }

  // drawer (phones)
  var menuBtn = document.querySelector(".menu-btn");
  var scrim = document.querySelector(".scrim");
  function setMenu(open) {
    if (!menuBtn) return;
    body.classList.toggle("menu-open", open);
    menuBtn.setAttribute("aria-expanded", String(open));
    menuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    if (scrim) scrim.hidden = !open;
  }
  if (menuBtn) menuBtn.addEventListener("click", function () { setMenu(!body.classList.contains("menu-open")); });
  if (scrim) scrim.addEventListener("click", function () { setMenu(false); });
  document.addEventListener("keydown", function (event) { if (event.key === "Escape") setMenu(false); });
  window.addEventListener("resize", function () { if (window.innerWidth > 767) setMenu(false); });

  // the All / Supported filter and the search boxes: one pass decides what shows
  var filterButtons = document.querySelectorAll("[data-filter]");
  var searches = document.querySelectorAll("[data-search]");
  function onlySupported() { return root.dataset.only === "supported"; }
  function refine(scope, rowSelector, query) {
    if (!scope) return;
    var only = onlySupported();
    var anyGroup = false;
    scope.querySelectorAll("[data-group]").forEach(function (group) {
      var shown = 0;
      group.querySelectorAll(rowSelector).forEach(function (row) {
        var hit = !query || (row.dataset.words || "").indexOf(query) !== -1;
        row.hidden = !hit;
        if (hit && (!only || row.hasAttribute("data-supported"))) shown++;
      });
      group.hidden = shown === 0;
      if (shown) anyGroup = true;
    });
    var empty = scope.querySelector(".empty, .sb-empty");
    if (empty) empty.hidden = anyGroup;
  }
  function apply() {
    var only = onlySupported();
    filterButtons.forEach(function (b) { b.setAttribute("aria-pressed", String((b.dataset.filter === "supported") === only)); });
    searches.forEach(function (input) {
      var query = input.value.trim().toLowerCase();
      if (input.dataset.search === "types") refine(document.getElementById("panel-types"), ".card", query);
      else refine(document.getElementById("sidebar"), "li", query);
    });
  }
  filterButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      var pick = button.dataset.filter;
      if (pick === "supported") root.dataset.only = "supported"; else delete root.dataset.only;
      store("rmv-filter", pick);
      apply();
    });
  });
  searches.forEach(function (input) {
    input.addEventListener("input", apply);
    input.addEventListener("keydown", function (event) {
      if (event.key !== "Enter") return;
      var scope = input.dataset.search === "types" ? document.getElementById("panel-types") : document.getElementById("sidebar");
      var first = scope && Array.prototype.find.call(scope.querySelectorAll("[data-supported]:not([hidden]) a"), function (a) {
        return !a.closest("[hidden]");
      });
      if (first) window.location.href = first.href;
    });
  });
  document.querySelectorAll("[data-reset]").forEach(function (button) {
    button.addEventListener("click", function () {
      searches.forEach(function (input) { input.value = ""; });
      delete root.dataset.only;
      store("rmv-filter", "all");
      apply();
    });
  });
  document.addEventListener("keydown", function (event) {
    var tag = (event.target.tagName || "").toLowerCase();
    if (event.key !== "/" || tag === "input" || tag === "textarea") return;
    var input = Array.prototype.find.call(searches, function (s) { return !s.closest("[hidden]"); });
    if (!input) return;
    event.preventDefault();
    if (input.closest("#sidebar") && window.innerWidth <= 767) setMenu(true);
    input.focus();
  });
  apply();
  var current = document.querySelector('.sidebar [aria-current="page"]');
  var sidebar = document.getElementById("sidebar");
  if (current && sidebar) sidebar.scrollTop = Math.max(0, current.offsetTop - sidebar.clientHeight / 2);

  // tabs (index): #types / #use-cases in the URL, arrow keys between them
  var tabs = Array.prototype.slice.call(document.querySelectorAll("[role=tab][data-tab]"));
  function selectTab(name, focus) {
    var tab = tabs.find(function (t) { return t.dataset.tab === name; });
    if (!tab) return;
    tabs.forEach(function (t) {
      var on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    document.querySelectorAll("[data-nav]").forEach(function (a) {
      if (a.dataset.nav === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (focus) tab.focus();
  }
  if (tabs.length) {
    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () {
        selectTab(tab.dataset.tab, false);
        history.replaceState(null, "", "#" + tab.dataset.tab);
      });
      tab.addEventListener("keydown", function (event) {
        var step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
        if (!step) return;
        event.preventDefault();
        var next = tabs[(i + step + tabs.length) % tabs.length];
        selectTab(next.dataset.tab, true);
        history.replaceState(null, "", "#" + next.dataset.tab);
      });
    });
    var fromHash = function () { selectTab(location.hash.slice(1) || "types", false); };
    window.addEventListener("hashchange", fromHash);
    fromHash();
  }

  // copy buttons
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(function () { return fallbackCopy(text); });
    }
    return fallbackCopy(text);
  }
  function fallbackCopy(text) {
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(area);
      if (ok) resolve(); else reject(new Error("copy refused"));
    });
  }
  document.querySelectorAll(".copy").forEach(function (button) {
    button.addEventListener("click", function () {
      var text = button.parentElement.querySelector("pre").innerText.replace(/\n$/, "");
      copyText(text).then(function () {
        button.textContent = "Copied";
        button.classList.add("done");
        setTimeout(function () { button.textContent = "Copy"; button.classList.remove("done"); }, 1400);
      }, function () {});
    });
  });

  // the player (entry pages): the still, the web viewer in place of it, the desktop command
  document.querySelectorAll(".player").forEach(function (player) {
    var stage = player.querySelector(".stage");
    var shot = player.querySelector(".stage-shot");
    var webButtons = player.querySelectorAll(".actions [data-act=web]");
    var notice = player.querySelector(".notice");
    var newtab = player.querySelector(".newtab");
    var served = location.protocol === "http:" || location.protocol === "https:";
    var rrdUrl = new URL(player.dataset.rrd, location.href).href;

    function viewerUrl(viewer) {
      // WHY persist=0: the hosted viewer otherwise remembers each app id's layout across visits, so an entry whose
      // blueprint changed would open in the stale one (robot_msg_viz.launch.web_url does the same).
      var url = viewer + (viewer.indexOf("?") === -1 ? "?" : "&") + "url=" + encodeURIComponent(rrdUrl) + "&persist=0";
      // A renderer pick on this page (?renderer=webgl) passes through: headless browsers have no WebGPU.
      var renderer = new URLSearchParams(location.search).get("renderer");
      return renderer ? url + "&renderer=" + encodeURIComponent(renderer) : url;
    }
    // WHY viewer.json: `serve` answers it with its local Rerun web viewer, and a site built with --viewer ships one
    // ("viewer/", relative to the site root); a static deploy without it uses the hosted app.rerun.io, which must be
    // at least as new as the SDK that wrote the .rrd.
    var viewerJson = new URL(base + "viewer.json", location.href).href;
    var viewer = !served ? Promise.resolve(null) : fetch(viewerJson, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (json) {
        return json && json.web_viewer ? new URL(json.web_viewer, viewerJson).href : player.dataset.webViewer;
      })
      .catch(function () { return player.dataset.webViewer; });
    viewer.then(function (url) {
      if (!url || !newtab) return;
      newtab.href = viewerUrl(url);
      newtab.hidden = false;
    });

    function say(text) { notice.textContent = text; notice.hidden = !text; }
    function setWebLabel(text) { webButtons.forEach(function (b) { b.querySelector("span").textContent = text; }); }
    function close() {
      var frame = stage.querySelector("iframe");
      if (frame) frame.remove();
      var wait = stage.querySelector(".wait");
      if (wait) wait.remove();
      stage.classList.remove("playing");
      player.classList.remove("playing");
      shot.hidden = false;
      setWebLabel("Open in browser");
    }
    function open() {
      if (!served) {
        say("The browser viewer fetches the recording over http, so this page must be served: " +
          "python -m robot_msg_viz.site serve <this folder>. Open on desktop works from here.");
        return;
      }
      say("");
      viewer.then(function (url) {
        shot.hidden = true;
        stage.classList.add("playing");
        player.classList.add("playing");
        var wait = document.createElement("p");
        wait.className = "wait";
        wait.textContent = "Loading the Rerun web viewer (about 30 MB the first time)…";
        stage.appendChild(wait);
        var frame = document.createElement("iframe");
        frame.src = viewerUrl(url);
        frame.title = "Rerun web viewer playing " + (player.dataset.title || "this recording");
        frame.allow = "fullscreen; clipboard-write";
        frame.addEventListener("load", function () { wait.remove(); });
        stage.appendChild(frame);
        setWebLabel("Back to the screenshot");
      });
    }
    player.querySelectorAll("[data-act=web]").forEach(function (button) {
      button.addEventListener("click", function () { if (stage.classList.contains("playing")) close(); else open(); });
    });

    var desktop = player.querySelector(".desktop");
    var desktopButton = player.querySelector("[data-act=desktop]");
    if (desktop && desktopButton) {
      desktopButton.addEventListener("click", function () {
        // From a file:// page the desktop viewer takes the file path itself. Behind a sign-in (data-desktop-file) the
        // viewer cannot fetch the URL, so the command opens the downloaded file and the download starts with it.
        var file = player.dataset.desktopFile;
        var target = file || (served ? rrdUrl : decodeURIComponent(new URL(rrdUrl).pathname));
        var command = player.dataset.desktop.replace("{url}", target);
        desktop.querySelector("[data-command] code").textContent = command;
        desktop.hidden = false;
        var heading = desktop.querySelector(".desktop-h");
        var prefix = "";
        if (file) {
          var link = player.querySelector("a[download]");
          if (link) link.click();
          prefix = "Downloading the .rrd (this page is behind a sign-in, so the viewer cannot fetch its URL). ";
        }
        copyText(command).then(function () { heading.textContent = prefix + "Copied to your clipboard. Paste it in a terminal:"; },
          function () { heading.textContent = prefix + "Run this in a terminal:"; });
      });
    }
  });
})();

(function () {

    // ── 建立全頁 drop overlay ────────────────────────────────
    var overlay = document.createElement('div');
    overlay.id = 'global-drop-overlay';
    overlay.style.cssText = [
        'display:none',
        'position:fixed',
        'inset:0',
        'z-index:9999',
        'background:rgba(13,110,253,0.12)',
        'backdrop-filter:blur(2px)',
        'pointer-events:none',
    ].join(';');

    var overlayBox = document.createElement('div');
    overlayBox.style.cssText = [
        'position:absolute',
        'top:50%',
        'left:50%',
        'transform:translate(-50%,-50%)',
        'border:3px dashed #0d6efd',
        'border-radius:16px',
        'padding:48px 64px',
        'background:rgba(255,255,255,0.92)',
        'text-align:center',
        'pointer-events:none',
        'box-shadow:0 8px 32px rgba(13,110,253,0.15)',
    ].join(';');
    overlayBox.innerHTML =
        '<div style="font-size:48px;margin-bottom:12px;">📂</div>' +
        '<div style="font-size:18px;font-weight:600;color:#0d6efd;">Drop .snp files anywhere to upload</div>';

    overlay.appendChild(overlayBox);

    function showOverlay() { overlay.style.display = 'block'; }
    function hideOverlay() { overlay.style.display = 'none'; }

    // ── 讀取所有檔案（sidebar drop zone 和全頁共用）────────
    function readFiles(files) {
        if (!files || !files.length) return;

        // 過濾非 snp 副檔名
        var snpExts = ['.s1p','.s2p','.s3p','.s4p','.s5p','.s6p','.snp'];
        var valid = Array.from(files).filter(function (f) {
            var ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
            return snpExts.indexOf(ext) !== -1;
        });
        if (!valid.length) return;

        var promises = valid.map(function (f) {
            return new Promise(function (resolve) {
                var reader = new FileReader();
                reader.onload = function (e) {
                    resolve({
                        content: e.target.result,
                        name: f.name,
                        size: Math.round(f.size / 1024 * 10) / 10
                    });
                };
                reader.readAsDataURL(f);
            });
        });

        Promise.all(promises).then(function (results) {
            window._pendingUpload = {
                contents:  results.map(function (r) { return r.content; }),
                filenames: results.map(function (r) { return r.name; }),
                sizes:     results.map(function (r) { return r.size; })
            };
        });
    }

    // ── 全頁 drag 事件（dragenter 計數法，避免子元素觸發 leave）
    var dragDepth = 0;

    document.addEventListener('dragenter', function (e) {
        e.preventDefault();
        dragDepth++;
        showOverlay();
    });

    document.addEventListener('dragover', function (e) {
        e.preventDefault();
    });

    document.addEventListener('dragleave', function (e) {
        dragDepth--;
        if (dragDepth <= 0) {
            dragDepth = 0;
            hideOverlay();
        }
    });

    document.addEventListener('drop', function (e) {
        e.preventDefault();
        dragDepth = 0;
        hideOverlay();
        readFiles(e.dataTransfer.files);
    });

    // ── 側邊欄 drop zone 設定 ────────────────────────────────
    function setupUploadZone() {
        var zone = document.getElementById('upload-drop-zone');
        if (!zone) return;
        if (zone.dataset.uploadReady) return;
        zone.dataset.uploadReady = '1';

        var inp = document.getElementById('file-input-native');
        if (!inp) {
            inp = document.createElement('input');
            inp.type     = 'file';
            inp.multiple = true;
            inp.accept   = '.s1p,.s2p,.s3p,.s4p,.s5p,.s6p,.snp';
            inp.id       = 'file-input-native';
            inp.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;cursor:pointer;z-index:1;';
            zone.style.position = 'relative';
            zone.appendChild(inp);
        }

        inp.addEventListener('change', function (e) {
            readFiles(e.target.files);
        });

        // sidebar zone 的 hover 樣式（全頁 drop 已處理實際檔案）
        inp.addEventListener('dragover', function (e) {
            e.preventDefault();
            zone.style.borderColor = '#0d6efd';
            zone.style.backgroundColor = '#eef2ff';
        });
        inp.addEventListener('dragleave', function (e) {
            if (!zone.contains(e.relatedTarget)) {
                zone.style.borderColor = '#ced4da';
                zone.style.backgroundColor = '#fff';
            }
        });
        inp.addEventListener('drop', function (e) {
            zone.style.borderColor = '#ced4da';
            zone.style.backgroundColor = '#fff';
            // 實際處理由全頁 drop 事件負責，這裡不重複呼叫
        });
    }

    // ── 初始化 ───────────────────────────────────────────────
    function init() {
        document.body.appendChild(overlay);

        var observer = new MutationObserver(function () {
            setupUploadZone();
        });
        observer.observe(document.body, { childList: true, subtree: true });
        setupUploadZone();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();

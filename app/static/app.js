/*
 * app/static/app.js
 * =================
 * LegalMind AI — chat UI over the persistent store.
 *
 * The browser holds NO history of its own. Every message rendered here came
 * out of the database via /api/conversations/<id>; `state` below is a cache
 * for the current paint, nothing more. That is what makes a hard refresh, a
 * new tab, or a server restart reproduce the conversation exactly.
 *
 * Identity and the active conversation live in the URL (?uid=…&c=…) so a
 * refresh lands in the same chat and the link can be reopened later.
 */

const state = {
    userId: null,
    conversationId: null,
    conversations: [],
    pendingUndo: null,     // { id, title, timer }
    searchMode: false,
    sending: false,
    // Analysis mode. The visible selector was removed, so this stays at the
    // former default: 'detailed' adds no instruction suffix to the query.
    mode: 'detailed',
};

/* ── uuid: crypto.randomUUID() only exists in a secure context, and this app
      is served over plain HTTP on the DGX, so a fallback is required. ───── */
function uuid() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        try { return window.crypto.randomUUID(); } catch (e) { /* fall through */ }
    }
    if (window.crypto && window.crypto.getRandomValues) {
        const b = window.crypto.getRandomValues(new Uint8Array(16));
        b[6] = (b[6] & 0x0f) | 0x40;
        b[8] = (b[8] & 0x3f) | 0x80;
        const hex = [...b].map(x => x.toString(16).padStart(2, '0')).join('');
        return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
    }
    return 'u-' + Date.now().toString(16) + '-' + Math.random().toString(16).slice(2, 10);
}

/* FastAPI's `detail` is a string for HTTPException but a list of
   {loc, msg} objects for request-validation (422) errors. Passing the list to
   `new Error()` rendered as "[object Object]", so flatten it to readable text. */
function errorDetailText(detail) {
    if (detail == null || detail === '') return '';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
        return detail.map(d => {
            if (typeof d === 'string') return d;
            const field = Array.isArray(d.loc) ? d.loc.filter(p => p !== 'body').join('.') : '';
            const msg = d.msg || d.message || JSON.stringify(d);
            return field ? `${field}: ${msg}` : msg;
        }).join('; ');
    }
    if (typeof detail === 'object') return detail.message || detail.msg || JSON.stringify(detail);
    return String(detail);
}

/* Where the API lives. Empty (the default) keeps every call relative, so the
   app talks to whatever host, port and scheme served the page — localhost,
   a shared server over plain http, or https. The backend injects
   window.LEGALMIND_API_BASE from the LEGALMIND_API_BASE environment variable;
   set it only when the API is on a different origin than the page (that
   origin then also needs LEGALMIND_CORS_ORIGINS on the backend). */
function apiUrl(path) {
    const base = (typeof window !== 'undefined' && window.LEGALMIND_API_BASE) || '';
    return base && path.startsWith('/') ? base.replace(/\/$/, '') + path : path;
}

async function api(path, options = {}) {
    const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData;
    const defaultHeaders = isFormData ? {} : (options.body ? { 'Content-Type': 'application/json' } : {});
    const headers = { ...defaultHeaders, ...(options.headers || {}) };
    const url = apiUrl(path);
    let res;
    try {
        res = await fetch(url, {
            credentials: 'same-origin',
            ...options,
            headers: headers,
        });
    } catch (err) {
        // fetch() rejects only when the request never completed: server down or
        // restarting, wrong host/port, blocked by CORS, or a bad certificate.
        // "Failed to fetch" alone says none of that, so name the target.
        if (err && err.name === 'AbortError') throw err;
        const target = new URL(url, window.location.href).origin;
        throw new Error(`Could not reach the server at ${target} (${err && err.message ? err.message : 'network error'}). `
                        + 'It may be restarting, unreachable from this browser, or blocking this origin.');
    }
    if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try { const j = await res.json(); detail = errorDetailText(j.detail) || detail; } catch (e) { /* keep */ }
        throw new Error(detail);
    }
    return res.status === 204 ? null : res.json();
}

window.fillPrompt = function(text) {
    const input = document.getElementById('query-input');
    if (!input) return;
    input.value = text;
    input.focus();
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 240) + 'px';
    const btn = document.getElementById('submit-btn');
    if (btn) btn.disabled = false;
};


/* ── URL <-> state ──────────────────────────────────────────────────────── */

function syncUrl() {
    const params = new URLSearchParams(window.location.search);
    if (state.userId) params.set('uid', state.userId);
    if (state.conversationId) params.set('c', state.conversationId);
    else params.delete('c');
    const next = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({ c: state.conversationId }, '', next);
}

/* ── Boot ───────────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', boot);

/* ── Shared activity feed ─────────────────────────────────────────────────
   The feed is global: every question from every user, on any device. It is
   read from the server on each load and polled, because another person's
   search happens in a different process with no way to push to this tab. */
const ACTIVITY_PAGE = 40;
const ACTIVITY_POLL_MS = 20000;

const feed = {
    tab: 'mine',          // 'mine' | 'everyone'
    offset: 0,
    total: 0,
    entries: [],
    readOnly: false,
    timer: null,
};

async function boot() {
    wireEvents();
    switchTab('mine');
    const params = new URLSearchParams(window.location.search);

    try {
        const me = await api('/api/me');
        state.userId = me.user_id;
    } catch (e) {
        state.userId = params.get('uid');
        console.error('Could not resolve identity:', e);
    }

    const wanted = params.get('c');
    await Promise.all([loadConversations(), loadActivity({ reset: true })]);
    startActivityPolling();

    if (wanted) {
        try {
            await openConversation(wanted, { allowPublic: true });
        } catch (e) {
            // A conversation id from someone else's browser, or a deleted one.
            showToast('That conversation is not available.', null);
            state.conversationId = null;
        }
    }
    syncUrl();
}

function wireEvents() {
    const form = document.getElementById('query-form');
    const input = document.getElementById('query-input');

    input.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
        if (this.value.trim() === '') this.style.height = '56px';
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (input.value.trim() !== '') form.requestSubmit();
        }
    });

    form.addEventListener('submit', onSubmit);

    document.getElementById('new-chat-btn').addEventListener('click', newChat);
    document.getElementById('conv-rename').addEventListener('click', () => renameActive());
    document.getElementById('conv-pin').addEventListener('click', () => togglePinActive());
    document.getElementById('conv-delete').addEventListener('click', () => deleteConversation(state.conversationId));
    document.getElementById('conv-export-md').addEventListener('click', () => exportActive('md'));
    document.getElementById('conv-export-txt').addEventListener('click', () => exportActive('txt'));

    document.getElementById('tab-mine').addEventListener('click', () => {
        switchTab('mine');
        const q = document.getElementById('history-search').value;
        if (q.trim()) runSearch(q);
        else {
            state.searchMode = false;
            renderConversationList();
        }
    });
    document.getElementById('tab-everyone').addEventListener('click', () => {
        switchTab('everyone');
        loadActivity({ reset: true });
    });

    const search = document.getElementById('history-search');
    const clearBtn = document.getElementById('search-clear-btn');
    let debounce = null;

    const updateClearBtn = () => {
        if (clearBtn) clearBtn.hidden = !(search.value && search.value.trim());
    };

    search.addEventListener('input', () => {
        updateClearBtn();
        clearTimeout(debounce);
        debounce = setTimeout(() => {
            if (feed.tab === 'everyone') loadActivity({ reset: true });
            else runSearch(search.value);
        }, 220);
    });

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            search.value = '';
            updateClearBtn();
            if (feed.tab === 'everyone') {
                loadActivity({ reset: true });
            } else {
                state.searchMode = false;
                renderConversationList();
            }
            search.focus();
        });
    }

    window.addEventListener('popstate', () => {
        const c = new URLSearchParams(window.location.search).get('c');
        if (c && c !== state.conversationId) openConversation(c);
        else if (!c) newChat();
    });

    // Mode Selector Pills
    document.querySelectorAll('.mode-pill').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-pill').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.mode = btn.dataset.mode || 'detailed';
        });
    });

    // Document & Image Attachment
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('doc-file-input');
    const preview = document.getElementById('attachment-preview');
    const removeBtn = document.getElementById('attachment-remove');

    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', async () => {
            const file = fileInput.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            uploadBtn.disabled = true;
            showToast('Uploading and analyzing legal document...', null);

            try {
                const res = await api('/api/upload', { method: 'POST', body: formData });
                state.attachedDoc = res;
                document.getElementById('attachment-name').textContent = res.filename;
                document.getElementById('attachment-size').textContent = `(${res.size_kb} KB)`;
                const isImg = /\.(png|jpg|jpeg|webp)$/i.test(res.filename);
                document.getElementById('attachment-icon').innerHTML = isImg ? '<i class="ri-image-line"></i>' : '<i class="ri-file-text-line"></i>';
                preview.hidden = false;
                showToast(`Attached ${res.filename} (${res.doc_type})`, null);
            } catch (err) {
                showToast(`Upload failed: ${err.message}`, null);
            } finally {
                uploadBtn.disabled = false;
                fileInput.value = '';
            }
        });
    }

    if (removeBtn) {
        removeBtn.addEventListener('click', () => {
            state.attachedDoc = null;
            if (preview) preview.hidden = true;
            showToast('Attachment removed', null);
        });
    }

    // Voice Typing (Web Speech API)
    const voiceBtn = document.getElementById('voice-btn');
    /* Browsers expose the microphone only in a "secure context": an https://
       origin, or http://localhost specifically. Served over plain http:// on
       a LAN IP or domain, SpeechRecognition fails with error "not-allowed"
       however the user answers the permission prompt — there is no in-page
       workaround, and getUserMedia/MediaRecorder are blocked the same way.
       So say what is actually wrong and where to go instead. */
    const secureUrl = () =>
        `https://${location.hostname}:${window.LEGALMIND_HTTPS_PORT || 8443}${location.pathname}${location.search}`;
    const micBlocked = () => !window.isSecureContext;
    if (voiceBtn) {
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) {
            voiceBtn.title = 'Voice typing not supported in this browser';
            voiceBtn.addEventListener('click', () => {
                showToast('Voice typing requires browser SpeechRecognition (e.g. Google Chrome/Edge).', null);
            });
        } else {
            if (!window.isSecureContext) {
                voiceBtn.classList.add('insecure');
                voiceBtn.title = 'Voice typing needs https — click for details';
            }
            const rec = new SpeechRec();
            rec.continuous = false;
            rec.interimResults = true;
            rec.lang = 'en-IN';

            rec.onstart = () => {
                voiceBtn.classList.add('listening');
                voiceBtn.innerHTML = '<i class="ri-mic-fill"></i>';
                showToast('Listening... Speak your legal inquiry now.', null);
            };

            rec.onresult = (e) => {
                let transcript = '';
                for (let i = e.resultIndex; i < e.results.length; i++) {
                    transcript += e.results[i][0].transcript;
                }
                input.value = transcript;
                input.style.height = 'auto';
                input.style.height = input.scrollHeight + 'px';
            };

            rec.onerror = (e) => {
                voiceBtn.classList.remove('listening');
                voiceBtn.innerHTML = '<i class="ri-mic-line"></i>';

                if ((e.error === 'not-allowed' || e.error === 'service-not-allowed') && micBlocked()) {
                    showToast(
                        `Microphone blocked: this page is served over plain HTTP, and browsers ` +
                        `only allow the mic on https:// or localhost. Open ${secureUrl()} instead.`,
                        () => { location.href = secureUrl(); }, 'Open secure site');
                    return;
                }
                const friendly = {
                    'not-allowed':         'Microphone permission was denied. Allow it in the address-bar site settings.',
                    'service-not-allowed': 'The browser refused the speech service. Check site permissions.',
                    'no-speech':           'No speech detected — try again and speak clearly.',
                    'audio-capture':       'No microphone was found on this device.',
                    'network':             'The speech service could not be reached.',
                    'aborted':             'Voice input was cancelled.',
                }[e.error];
                showToast(friendly || `Voice input error: ${e.error}`, null);
            };

            rec.onend = () => {
                voiceBtn.classList.remove('listening');
                voiceBtn.innerHTML = '<i class="ri-mic-line"></i>';
                if (input.value.trim()) {
                    input.focus();
                }
            };

            voiceBtn.addEventListener('click', () => {
                if (micBlocked()) {
                    showToast(
                        `Voice input needs a secure (https) connection. This page is on ` +
                        `${location.protocol}//${location.host}. Open ${secureUrl()} instead.`,
                        () => { location.href = secureUrl(); }, 'Open secure site');
                    return;
                }
                if (voiceBtn.classList.contains('listening')) {
                    rec.stop();
                } else {
                    rec.start();
                }
            });
        }
    }

    initScrollListeners();
}

/* ── Submitting a question ──────────────────────────────────────────────── */

async function onSubmit(e) {
    e.preventDefault();
    if (state.sending) return;

    const input = document.getElementById('query-input');
    const query = input.value.trim();
    if (!query) return;

    /* One token per submit. The server keys on it, so a retry, a double
       click or a resubmitted form can never create a second row — and never
       runs the model twice. Position in the script has nothing to do with it. */
    const clientToken = uuid();

    state.sending = true;
    input.value = '';
    input.style.height = '56px';
    document.getElementById('submit-btn').disabled = true;
    // If currently viewing a read-only shared conversation from someone else,
    // detach and start a new conversation for this user.
    if (feed.readOnly) {
        state.conversationId = null;
        feed.readOnly = false;
        switchTab('mine');
    }

    userScrolledUp = false;
    updateScrollBottomButton();

    appendUserMessage({ content: query, status: 'complete' });
    const pending = appendBotLoading();
    scrollToBottom(true);

    // Prepare full query with attached document or mode instructions
    let fullQuery = query;
    if (state.attachedDoc && state.attachedDoc.extracted_text) {
        fullQuery = `[ATTACHED DOCUMENT: ${state.attachedDoc.filename || 'Document'} (${state.attachedDoc.doc_type || 'File'})]\n"""\n${state.attachedDoc.extracted_text}\n"""\n\n[USER QUERY]\n${query}`;
        state.attachedDoc = null;
        const prev = document.getElementById('attachment-preview');
        if (prev) prev.hidden = true;
    }

    if (state.mode === 'simple') {
        fullQuery += "\n\n(Please explain in simple, plain English accessible to a non-lawyer, breaking down legal terms clearly.)";
    } else if (state.mode === 'student') {
        fullQuery += "\n\n(Structure the response for a law student/exam candidate: state the legal issue, relevant statutory provisions, leading case ratios, and step-by-step judicial analysis.)";
    } else if (state.mode === 'case-analysis') {
        fullQuery += "\n\n(Focus specifically on case law and landmark precedents: cite relevant Supreme Court / High Court decisions, their bench corams, ratios, and subsequent treatment.)";
    }

    try {
        let streamSucceeded = false;
        try {
            const streamRes = await fetch(apiUrl('/api/chat/stream'), {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: fullQuery,
                    conversation_id: state.conversationId,
                    client_token: clientToken,
                }),
            });

            if (streamRes.ok && (streamRes.headers.get('content-type') || '').includes('text/event-stream')) {
                const reader = streamRes.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';
                let accumulatedText = '';
                const payloadEl = pending.querySelector('.answer-payload');
                const loadingEl = pending.querySelector('.loading-state');
                const markdownEl = payloadEl.querySelector('.markdown-body');
                let finalData = null;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    let currentEvent = 'message';
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed) continue;
                        if (trimmed.startsWith('event:')) {
                            currentEvent = trimmed.slice(6).trim();
                        } else if (trimmed.startsWith('data:')) {
                            const rawData = trimmed.slice(5).trim();
                            if (!rawData) continue;
                            try {
                                const parsed = JSON.parse(rawData);
                                if (currentEvent === 'meta' || currentEvent === 'metadata') {
                                    if (parsed.conversation_id) {
                                        state.conversationId = parsed.conversation_id;
                                        syncUrl();
                                    }
                                    if (loadingEl) {
                                        const lt = loadingEl.querySelector('.loading-text');
                                        if (lt) lt.textContent = 'Generating legal analysis...';
                                    }
                                } else if (currentEvent === 'token') {
                                    if (loadingEl && loadingEl.style.display !== 'none') {
                                        loadingEl.style.display = 'none';
                                        payloadEl.style.display = 'block';
                                    }
                                    const chunk = parsed.delta || parsed.token || '';
                                    accumulatedText += chunk;
                                    if (typeof marked !== 'undefined') {
                                        markdownEl.innerHTML = marked.parse(accumulatedText);
                                    } else {
                                        markdownEl.textContent = accumulatedText;
                                    }
                                    autoScrollDuringStream();
                                } else if (currentEvent === 'done') {
                                    finalData = parsed;
                                } else if (currentEvent === 'error') {
                                    throw new Error(parsed.error || 'Streaming error');
                                }
                            } catch (parseErr) {
                                if (currentEvent === 'error') throw parseErr;
                            }
                        }
                    }
                }

                if (finalData && finalData.assistant_message) {
                    fillBotMessage(pending, finalData.assistant_message);
                    setConversationBar(finalData.conversation);
                    await Promise.all([loadConversations(), loadActivity({ reset: true, silent: true })]);
                    streamSucceeded = true;
                }
            }
        } catch (streamErr) {
            console.warn('[Stream] Fallback to /api/chat due to:', streamErr);
        }

        if (!streamSucceeded) {
            const data = await api('/api/chat', {
                method: 'POST',
                body: JSON.stringify({
                    query: fullQuery,
                    conversation_id: state.conversationId,
                    client_token: clientToken,
                }),
            });

            state.conversationId = data.conversation_id;
            syncUrl();

            if (data.assistant_message) {
                fillBotMessage(pending, data.assistant_message);
            } else {
                fillBotError(pending, data.error || 'No answer was recorded.');
            }
            setConversationBar(data.conversation);
            // Refresh both lists: your question belongs in the shared feed too.
            await Promise.all([loadConversations(), loadActivity({ reset: true, silent: true })]);
        }
    } catch (err) {
        /* The question itself is already committed server-side, so a failure
           here costs the answer, never the question — a reload shows it. */
        fillBotError(pending, err.message);
        await loadConversations();
    } finally {
        state.sending = false;
        document.getElementById('submit-btn').disabled = false;
        if (!userScrolledUp) {
            setTimeout(() => scrollToBottom(false), 80);
        }
        updateScrollBottomButton();
    }
}

/* ── Rendering messages (always from stored rows) ───────────────────────── */

function clearTranscript() {
    document.getElementById('chat-history').innerHTML = '';
}

function appendUserMessage(msg) {
    const clone = document.getElementById('msg-user-template').content.cloneNode(true);
    clone.querySelector('.text').textContent = msg.content;
    document.getElementById('chat-history').appendChild(clone);
}

function appendBotLoading() {
    const clone = document.getElementById('msg-bot-template').content.cloneNode(true);
    const el = clone.querySelector('.message');
    document.getElementById('chat-history').appendChild(clone);
    return el;
}

function fillBotMessage(el, msg) {
    el.querySelector('.loading-state').style.display = 'none';
    const payload = el.querySelector('.answer-payload');
    payload.style.display = 'block';

    const meta = msg.metadata || {};
    const body = payload.querySelector('.markdown-body');
    body.innerHTML = marked.parse(msg.content || '');

    if (msg.status && msg.status !== 'complete') {
        const note = document.createElement('p');
        note.className = 'status-note ' + msg.status;
        note.textContent = msg.status === 'partial'
            ? 'This answer is incomplete — generation was interrupted. The question is saved.'
            : 'This answer failed to generate' + (meta.error ? `: ${meta.error}` : '.');
        body.prepend(note);
    }

    const metrics = payload.querySelector('.metrics');
    const rLat = meta.retrieval_latency, gLat = meta.generation_latency;
    if (rLat == null && gLat == null) {
        metrics.style.display = 'none';
    } else {
        payload.querySelector('.r-lat').textContent = rLat != null ? rLat : '—';
        payload.querySelector('.g-lat').textContent = gLat != null ? gLat : '—';
    }

    /* Confidence is shown ONLY when the support score has been demonstrated to
       track correctness (see evaluation/calibrate_confidence.py). Until then we
       show what was actually verified instead of a confidence word — an
       uncalibrated "HIGH" on a legal answer is worse than no number at all. */
    const badge = payload.querySelector('.confidence-badge');
    const conf = (meta.confidence || '').toString();
    const verification = meta.verification || null;

    if (conf && meta.confidence_calibrated) {
        const lower = conf.toLowerCase();
        badge.textContent = 'Confidence: ' + conf.split('\n')[0].trim();
        badge.className = 'confidence-badge ' + (
            lower.includes('high') ? 'conf-high' : lower.includes('medium') ? 'conf-medium' : 'conf-low');
    } else if (verification && verification.checked) {
        const fab = verification.unverified || 0;
        badge.textContent = fab
            ? `${fab} citation${fab === 1 ? '' : 's'} unverified`
            : `${verification.grounded}/${verification.adjudicable || verification.checked} citations verified`;
        badge.className = 'confidence-badge ' + (fab ? 'conf-low' : 'conf-high');
        badge.title = meta.evidence_summary || '';
    } else if (meta.evidence_summary) {
        badge.textContent = meta.evidence_summary;
        badge.className = 'confidence-badge conf-medium';
    } else {
        badge.style.display = 'none';
    }

    /* Anything the answer asserted that our sources could not confirm is
       surfaced next to the answer, not buried in a tooltip. */
    if (verification && verification.unverified) {
        const bad = (verification.findings || []).filter(f => f.status === 'unverified');
        if (bad.length) {
            const warn = document.createElement('p');
            warn.className = 'status-note error';
            warn.textContent = '⚠ Could not be verified against our source judgments: '
                + bad.map(f => f.text).slice(0, 6).join(', ')
                + '. Check the official reports before relying on these.';
            payload.querySelector('.markdown-body').appendChild(warn);
        }
    }

    renderAuthorities(payload, msg.citations || []);
    const disc = payload.querySelector('.bot-disclaimer');
    if (disc) disc.style.display = 'flex';
}

function fillBotError(el, message) {
    el.querySelector('.loading-state').style.display = 'none';
    const payload = el.querySelector('.answer-payload');
    payload.style.display = 'block';
    payload.querySelector('.markdown-body').innerHTML =
        `<p class="status-note error"><i class="ri-error-warning-line"></i> ${escapeHtml(message)}</p>`;
    payload.querySelector('.metrics').style.display = 'none';
    payload.querySelector('.sources-accordion').style.display = 'none';
    const disc = payload.querySelector('.bot-disclaimer');
    if (disc) disc.style.display = 'none';
}

function renderAuthorities(payload, citations) {
    const accordion = payload.querySelector('.sources-accordion');
    if (!citations.length) { accordion.style.display = 'none'; return; }

    payload.querySelector('.src-count').textContent = citations.length;
    const list = payload.querySelector('.source-list');
    list.innerHTML = '';

    citations.forEach((cite, i) => {
        const li = document.createElement('li');
        li.className = 'source-item';

        const tag = document.createElement('span');
        tag.className = 'src-tag';
        tag.textContent = `[${cite.citation_id || i + 1}]`;
        li.appendChild(tag);

        const title = document.createElement('span');
        title.className = 'src-title';
        title.textContent = cite.case_name || cite.citation || 'Unattributed source';
        li.appendChild(title);

        const bits = [cite.citation, cite.court, cite.date, cite.judges,
                      cite.paragraph_no ? `Para ${cite.paragraph_no}` : null,
                      cite.source_id ? `Source: ${cite.source_id}` : null]
                      .filter(Boolean);
        if (bits.length) {
            const m = document.createElement('div');
            m.className = 'src-meta';
            m.textContent = bits.join(' | ');
            li.appendChild(m);
        }
        if (cite.excerpt) {
            const t = document.createElement('div');
            t.className = 'src-text';
            t.textContent = `"${cite.excerpt}…"`;
            li.appendChild(t);
        }
        list.appendChild(li);
    });
}

/* ── Opening a stored conversation ──────────────────────────────────────── */

async function openConversation(id, opts = {}) {
    let data;
    let readOnly = false;
    try {
        data = await api(`/api/conversations/${encodeURIComponent(id)}`);
        readOnly = !!data.read_only;
    } catch (err) {
        /* Not this browser's conversation. If it came from the shared feed it
           is still readable through the public endpoint — look, don't touch. */
        try {
            data = await api(`/api/activity/${encodeURIComponent(id)}`);
            readOnly = true;
        } catch (e2) {
            if (!opts.allowPublic) throw err;
            throw e2;
        }
    }
    feed.readOnly = readOnly;
    state.conversationId = data.conversation.id;

    clearTranscript();
    switchSuite('research');
    document.getElementById('hero-section').classList.add('shrunk');

    (data.messages || []).forEach(msg => {
        if (msg.role === 'user') {
            appendUserMessage(msg);
        } else {
            fillBotMessage(appendBotLoading(), msg);
        }
    });

    setConversationBar(data.conversation);
    markActiveRow();
    syncUrl();
    setTimeout(scrollToBottom, 50);
    return data;
}

function newChat() {
    state.conversationId = null;
    clearTranscript();
    switchSuite('research');
    document.getElementById('hero-section').classList.remove('shrunk');
    document.getElementById('conv-bar').hidden = true;
    markActiveRow();
    syncUrl();
    const input = document.getElementById('query-input');
    if (input) input.focus();
}

function setConversationBar(conv) {
    const bar = document.getElementById('conv-bar');
    if (!conv || !conv.id) { bar.hidden = true; return; }
    bar.hidden = false;
    document.getElementById('conv-title-display').textContent = conv.title || 'Untitled';
    const pin = document.getElementById('conv-pin');
    pin.classList.toggle('active', !!conv.pinned);
    pin.title = conv.pinned ? 'Unpin' : 'Pin';

    /* Somebody else's conversation, opened from the shared feed: the server
       refuses these mutations anyway, so don't offer them. */
    document.getElementById('readonly-badge').hidden = !feed.readOnly;
    ['conv-rename', 'conv-pin', 'conv-delete', 'conv-export-md', 'conv-export-txt']
        .forEach(id => { document.getElementById(id).hidden = feed.readOnly; });
}

/* ── Shared activity feed ───────────────────────────────────────────────── */

function switchTab(tab) {
    feed.tab = tab;
    document.getElementById('tab-mine').classList.toggle('is-active', tab === 'mine');
    document.getElementById('tab-everyone').classList.toggle('is-active', tab === 'everyone');
    document.getElementById('tab-mine').setAttribute('aria-selected', tab === 'mine');
    document.getElementById('tab-everyone').setAttribute('aria-selected', tab === 'everyone');

    const convList = document.getElementById('conversation-list');
    const actList = document.getElementById('activity-list');
    
    convList.hidden = tab !== 'mine';
    convList.style.display = tab === 'mine' ? 'flex' : 'none';

    actList.hidden = tab !== 'everyone';
    actList.style.display = tab === 'everyone' ? 'flex' : 'none';

    const searchInput = document.getElementById('history-search');
    searchInput.placeholder =
        tab === 'everyone' ? 'Search everyone\u2019s searches\u2026' : 'Search your history\u2026';
    const clearBtn = document.getElementById('search-clear-btn');
    if (clearBtn) clearBtn.hidden = !(searchInput.value && searchInput.value.trim());
    renderFoot();
}

async function loadActivity({ reset = false, silent = false } = {}) {
    if (reset) { feed.offset = 0; feed.entries = []; }
    const q = document.getElementById('history-search').value.trim();
    const params = new URLSearchParams({ limit: ACTIVITY_PAGE, offset: feed.offset });
    if (q && feed.tab === 'everyone') params.set('q', q);

    try {
        const data = await api(`/api/activity?${params}`);
        feed.entries = reset ? data.entries : feed.entries.concat(data.entries);
        feed.total = data.total;
        feed.offset = feed.entries.length;
        feed.hasMore = data.has_more;
    } catch (e) {
        if (!silent) {
            document.getElementById('activity-list').innerHTML =
                '<p class="sidebar-empty">Shared history unavailable.</p>';
        }
        return;
    }
    renderActivity();
    renderFoot();
}

/* Another person's search lands in a different process, so there is nothing
   to push into this tab — poll instead, and only while the tab is visible. */
function startActivityPolling() {
    stopActivityPolling();
    feed.timer = setInterval(() => {
        if (document.hidden) return;
        if (feed.tab !== 'everyone') return;
        loadActivity({ reset: true, silent: true });
    }, ACTIVITY_POLL_MS);
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && feed.tab === 'everyone') {
            loadActivity({ reset: true, silent: true });
        }
    });
}

function stopActivityPolling() {
    if (feed.timer) clearInterval(feed.timer);
    feed.timer = null;
}

function renderActivity() {
    const box = document.getElementById('activity-list');
    box.innerHTML = '';

    if (!feed.entries.length) {
        box.innerHTML = '<p class="sidebar-empty">No searches yet. Ask something — everyone will see it here.</p>';
        return;
    }

    let lastGroup = null;
    feed.entries.forEach(entry => {
        const group = dayBucket(entry.created_at);
        if (group !== lastGroup) {
            const h = document.createElement('div');
            h.className = 'sidebar-group';
            h.textContent = group;
            box.appendChild(h);
            lastGroup = group;
        }

        const item = document.createElement('button');
        item.className = 'activity-item';
        item.dataset.id = entry.conversation_id;
        if (entry.conversation_id === state.conversationId) item.classList.add('is-active');

        const dot = document.createElement('span');
        dot.className = 'dot ' + (!entry.answered ? 'pending' : (entry.answer_status || 'complete'));
        dot.title = !entry.answered ? 'awaiting answer' : `answer: ${entry.answer_status}`;
        item.appendChild(dot);

        const body = document.createElement('span');
        body.className = 'activity-body';

        const q = document.createElement('span');
        q.className = 'activity-q';
        q.textContent = entry.query;
        body.appendChild(q);

        const meta = document.createElement('span');
        meta.className = 'activity-meta';
        const who = document.createElement('span');
        who.className = 'who';
        who.textContent = entry.who === state.userId ? 'you' : entry.who;
        meta.appendChild(who);
        meta.appendChild(document.createTextNode(' \u00b7 ' + shortTime(entry.created_at)));
        body.appendChild(meta);

        item.appendChild(body);
        item.addEventListener('click', () => openConversation(entry.conversation_id, { allowPublic: true }));
        box.appendChild(item);
    });

    if (feed.hasMore) {
        const more = document.createElement('button');
        more.className = 'load-more';
        more.textContent = `Load older (${feed.total - feed.entries.length} more)`;
        more.addEventListener('click', async () => {
            more.disabled = true;
            more.textContent = 'Loading\u2026';
            await loadActivity();
        });
        box.appendChild(more);
    }
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */

async function loadConversations() {
    try {
        const data = await api('/api/conversations?limit=200');
        state.userId = data.user_id || state.userId;
        state.conversations = data.conversations || [];
    } catch (e) {
        document.getElementById('conversation-list').innerHTML =
            '<p class="sidebar-empty">History unavailable.</p>';
        return;
    }
    if (!state.searchMode) renderConversationList();
    renderFoot();
}

function dayBucket(iso) {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return 'Earlier';

    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startOfYesterday = new Date(startOfToday);
    startOfYesterday.setDate(startOfYesterday.getDate() - 1);
    const startOf7Days = new Date(startOfToday);
    startOf7Days.setDate(startOf7Days.getDate() - 7);

    if (d >= startOfToday) return 'Today';
    if (d >= startOfYesterday) return 'Yesterday';
    if (d >= startOf7Days) return 'Previous 7 days';
    return 'Earlier';
}

function renderConversationList() {
    const container = document.getElementById('conversation-list');
    container.innerHTML = '';

    if (!state.conversations || !state.conversations.length) {
        container.innerHTML = '<p class="sidebar-empty">No conversations yet. Ask something to start one.</p>';
        return;
    }

    // Sort conversations: pinned first, then newest updated_at timestamp first
    const sorted = [...state.conversations].sort((a, b) => {
        if (!!a.pinned !== !!b.pinned) return a.pinned ? -1 : 1;
        const timeA = new Date(a.updated_at || 0).getTime();
        const timeB = new Date(b.updated_at || 0).getTime();
        return timeB - timeA;
    });

    const pinned = sorted.filter(c => c.pinned);
    const rest = sorted.filter(c => !c.pinned);

    const groups = [];
    if (pinned.length) groups.push(['Pinned', pinned]);
    for (const label of ['Today', 'Yesterday', 'Previous 7 days', 'Earlier']) {
        const items = rest.filter(c => dayBucket(c.updated_at) === label);
        if (items.length) groups.push([label, items]);
    }

    groups.forEach(([label, items]) => {
        const h = document.createElement('div');
        h.className = 'sidebar-group';
        h.textContent = label;
        container.appendChild(h);
        items.forEach(conv => container.appendChild(conversationRow(conv)));
    });
    markActiveRow();
}

function conversationRow(conv) {
    const clone = document.getElementById('conv-row-template').content.cloneNode(true);
    const row = clone.querySelector('.conv-row');
    row.dataset.id = conv.id;

    const titleText = ((conv && conv.title) ? String(conv.title) : 'Untitled').trim().replace(/[\r\n\t]+/g, ' ') || 'Untitled';
    const titleEl = row.querySelector('.conv-row-title');
    titleEl.textContent = titleText;
    titleEl.title = titleText;

    const count = (conv && conv.message_count) || 0;
    const metaEl = row.querySelector('.conv-row-meta');
    const metaText = `${count} message${count === 1 ? '' : 's'} · ${shortTime(conv ? conv.updated_at : '')}`;
    metaEl.textContent = metaText;
    metaEl.title = metaText;

    const openBtn = row.querySelector('.conv-open');
    openBtn.title = `${titleText} (${metaText})`;
    openBtn.addEventListener('click', () => openConversation(conv.id));

    const pinBtn = row.querySelector('.pin');
    pinBtn.classList.toggle('active', !!(conv && conv.pinned));
    pinBtn.title = (conv && conv.pinned) ? 'Unpin' : 'Pin';
    pinBtn.addEventListener('click', (e) => { e.stopPropagation(); togglePin(conv); });

    row.querySelector('.rename').addEventListener('click', (e) => { e.stopPropagation(); renameConversation(conv); });
    row.querySelector('.delete').addEventListener('click', (e) => { e.stopPropagation(); deleteConversation(conv.id); });
    return row;
}

function markActiveRow() {
    document.querySelectorAll('.conv-row').forEach(row => {
        row.classList.toggle('active', row.dataset.id === state.conversationId);
    });
}

function renderFoot() {
    const foot = document.getElementById('sidebar-foot');
    if (feed.tab === 'everyone') {
        foot.textContent =
            `${feed.total} search(es) from everyone · shared across all devices`;
        return;
    }
    const total = state.conversations.reduce((n, c) => n + (c.message_count || 0), 0);
    const uid = state.userId ? state.userId.slice(0, 8) : '—';
    foot.textContent = `${state.conversations.length} conversation(s) · ${total} message(s) · browser ${uid}`;
}

function shortTime(iso) {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const bucket = dayBucket(iso);
    if (bucket === 'Today') return timeStr;
    if (bucket === 'Yesterday') return `Yesterday · ${timeStr}`;
    const dateStr = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    return `${dateStr} · ${timeStr}`;
}

/* ── Rename / pin / delete + undo ───────────────────────────────────────── */

async function renameConversation(conv) {
    const next = window.prompt('Rename conversation', conv.title || '');
    if (next === null) return;
    if (!next.trim()) { showToast('A title cannot be empty.', null); return; }
    await api(`/api/conversations/${encodeURIComponent(conv.id)}`, {
        method: 'PATCH',
        body: JSON.stringify({ title: next.trim() }),
    });
    await loadConversations();
    if (state.conversationId === conv.id) {
        setConversationBar(state.conversations.find(c => c.id === conv.id));
    }
}

function renameActive() {
    const conv = state.conversations.find(c => c.id === state.conversationId);
    if (conv) renameConversation(conv);
}

async function togglePin(conv) {
    await api(`/api/conversations/${encodeURIComponent(conv.id)}`, {
        method: 'PATCH',
        body: JSON.stringify({ pinned: !conv.pinned }),
    });
    await loadConversations();
    if (state.conversationId === conv.id) {
        setConversationBar(state.conversations.find(c => c.id === conv.id));
    }
}

function togglePinActive() {
    const conv = state.conversations.find(c => c.id === state.conversationId);
    if (conv) togglePin(conv);
}

async function deleteConversation(id) {
    if (!id) return;
    const conv = state.conversations.find(c => c.id === id);
    await api(`/api/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (state.conversationId === id) newChat();
    await loadConversations();
    /* Soft delete: the rows are still there, so undo is a restore call. */
    showToast(`Deleted "${conv ? conv.title : 'conversation'}".`, async () => {
        await api(`/api/conversations/${encodeURIComponent(id)}/restore`, { method: 'POST' });
        await loadConversations();
        await openConversation(id);
    });
}

function showToast(text, undoAction, actionLabel) {
    const toast = document.getElementById('toast');
    const action = document.getElementById('toast-action');
    document.getElementById('toast-text').textContent = text;
    if (state.pendingUndo) clearTimeout(state.pendingUndo.timer);

    // The action button is reused for undo and for other one-shot actions
    // (e.g. "Open secure site"), so its label is caller-supplied.
    action.hidden = !undoAction;
    action.textContent = actionLabel || 'Undo';
    action.onclick = undoAction ? async () => {
        toast.hidden = true;
        clearTimeout(state.pendingUndo.timer);
        state.pendingUndo = null;
        await undoAction();
    } : null;

    toast.hidden = false;
    state.pendingUndo = { timer: setTimeout(() => { toast.hidden = true; }, 9000) };
}

/* ── Export ─────────────────────────────────────────────────────────────── */

function exportActive(fmt) {
    if (!state.conversationId) return;
    /* A normal navigation so the browser saves the file under the name the
       server sets in Content-Disposition (<date>-<slug>.md). */
    window.location.href =
        `/api/conversations/${encodeURIComponent(state.conversationId)}/export?format=${fmt}`;
}

/* ── History search (FTS5) ──────────────────────────────────────────────── */

async function runSearch(raw) {
    const q = (raw || '').trim();
    if (!q) {
        state.searchMode = false;
        renderConversationList();
        return;
    }
    state.searchMode = true;
    const container = document.getElementById('conversation-list');
    try {
        const data = await api(`/api/search?q=${encodeURIComponent(q)}`);
        const hits = data.hits || [];

        // Check local conversation titles matching query
        const titleMatches = (state.conversations || []).filter(c =>
            (c.title || '').toLowerCase().includes(q.toLowerCase())
        );

        const matchedConvIds = new Set([
            ...titleMatches.map(c => c.id),
            ...hits.map(h => h.conversation_id)
        ]);

        container.innerHTML = '';
        const head = document.createElement('div');
        head.className = 'sidebar-group';
        const total = matchedConvIds.size;
        head.textContent = `${total} result${total === 1 ? '' : 's'} for "${q}"`;
        container.appendChild(head);

        if (total === 0) {
            container.insertAdjacentHTML('beforeend',
                '<p class="sidebar-empty">Nothing matched.</p>');
            return;
        }

        const renderList = (state.conversations || []).filter(c => matchedConvIds.has(c.id));
        if (renderList.length > 0) {
            renderList.sort((a, b) => new Date(b.updated_at || 0).getTime() - new Date(a.updated_at || 0).getTime());
            renderList.forEach(conv => container.appendChild(conversationRow(conv)));
        } else {
            hits.forEach(hit => {
                const row = document.createElement('div');
                row.className = 'conv-row search-hit';
                const btn = document.createElement('button');
                btn.className = 'conv-open';
                const t = document.createElement('span');
                t.className = 'conv-row-title';
                t.textContent = hit.conversation_title || 'Untitled';
                const s = document.createElement('span');
                s.className = 'conv-row-snippet';
                s.innerHTML = escapeHtml(hit.snippet).replace(/&lt;mark&gt;/g, '<mark>')
                                                     .replace(/&lt;\/mark&gt;/g, '</mark>');
                btn.appendChild(t);
                btn.appendChild(s);
                btn.addEventListener('click', () => openConversation(hit.conversation_id));
                row.appendChild(btn);
                container.appendChild(row);
            });
        }
    } catch (e) {
        container.innerHTML = '<p class="sidebar-empty">Search failed.</p>';
    }
}

/* ── Misc ───────────────────────────────────────────────────────────────── */

function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s == null ? '' : String(s);
    return div.innerHTML;
}

function toggleSources(button) {
    button.classList.toggle('active');
    button.nextElementSibling.classList.toggle('active');
}

/* ── Smart Viewport & Auto-Scroll Controller ────────────────────────────── */

let userScrolledUp = false;

function isUserNearBottom(threshold = 120) {
    const history = document.getElementById('chat-history');
    const page = document.scrollingElement || document.documentElement;
    
    const pageDist = page ? (page.scrollHeight - page.scrollTop - window.innerHeight) : 0;
    const historyDist = history ? (history.scrollHeight - history.scrollTop - history.clientHeight) : 0;
    
    if (page && page.scrollHeight > window.innerHeight + 20 && pageDist > threshold) return false;
    if (history && history.scrollHeight > history.clientHeight + 20 && historyDist > threshold) return false;
    
    return true;
}

function updateScrollBottomButton() {
    const btn = document.getElementById('scroll-bottom-btn');
    if (!btn) return;
    if (userScrolledUp || !isUserNearBottom(90)) {
        btn.classList.add('visible');
    } else {
        btn.classList.remove('visible');
    }
}

window.userScrollToBottom = function () {
    userScrolledUp = false;
    const btn = document.getElementById('scroll-bottom-btn');
    if (btn) btn.classList.remove('visible');
    scrollToBottom(true);
};

function scrollToBottom(smooth = true) {
    const history = document.getElementById('chat-history');
    const page = document.scrollingElement || document.documentElement;
    const behavior = smooth ? 'smooth' : 'auto';
    
    if (history && history.scrollHeight > history.clientHeight) {
        history.scrollTo({ top: history.scrollHeight, behavior });
    }
    if (page && page.scrollHeight > window.innerHeight) {
        page.scrollTo({ top: page.scrollHeight, behavior });
    }
}

function autoScrollDuringStream() {
    // If the user has intentionally scrolled up to view top sections, do NOT force scroll down!
    if (userScrolledUp) {
        updateScrollBottomButton();
        return;
    }
    
    if (isUserNearBottom(140)) {
        // Fast instant scroll to follow incoming stream tokens
        scrollToBottom(false);
        updateScrollBottomButton();
    } else {
        // User naturally scrolled up
        userScrolledUp = true;
        updateScrollBottomButton();
    }
}

function initScrollListeners() {
    const history = document.getElementById('chat-history');
    const onScrollChange = () => {
        if (isUserNearBottom(90)) {
            userScrolledUp = false;
        } else {
            userScrolledUp = true;
        }
        updateScrollBottomButton();
    };

    window.addEventListener('wheel', (e) => {
        if (e.deltaY < -2) {
            // User scrolled UP
            userScrolledUp = true;
            updateScrollBottomButton();
        } else if (e.deltaY > 2 && isUserNearBottom(80)) {
            // User reached bottom
            userScrolledUp = false;
            updateScrollBottomButton();
        }
    }, { passive: true });

    let touchStartY = 0;
    window.addEventListener('touchstart', (e) => {
        if (e.touches && e.touches[0]) touchStartY = e.touches[0].clientY;
    }, { passive: true });

    window.addEventListener('touchmove', (e) => {
        if (e.touches && e.touches[0]) {
            const currentY = e.touches[0].clientY;
            if (currentY > touchStartY + 10) {
                // Swiped down -> scrolled UP
                userScrolledUp = true;
                updateScrollBottomButton();
            } else if (isUserNearBottom(80)) {
                userScrolledUp = false;
                updateScrollBottomButton();
            }
        }
    }, { passive: true });

    window.addEventListener('scroll', onScrollChange, { passive: true });
    if (history) {
        history.addEventListener('scroll', onScrollChange, { passive: true });
    }
}

/* ── Listen: read a research answer aloud ─────────────────────────────────
   The answer template calls toggleSpeech(this), but nothing ever defined it,
   so every click threw a ReferenceError. This speaks the answer text in the
   browser with the Web Speech API — no server call. One answer speaks at a
   time: clicking the playing button stops it, clicking another one switches. */
const answerSpeech = { btn: null, run: 0 };
const SPEECH_LANGS = { en: 'en-IN', hi: 'hi-IN', bn: 'bn-IN', ta: 'ta-IN', te: 'te-IN',
                       mr: 'mr-IN', gu: 'gu-IN', kn: 'kn-IN', ml: 'ml-IN', pa: 'pa-IN', ur: 'ur-IN' };

function resetListenButton(btn) {
    if (!btn) return;
    btn.classList.remove('speaking');
    btn.innerHTML = '<i class="ri-volume-up-line"></i> <span>Listen</span>';
    btn.title = 'Listen to Answer (Text-to-Speech)';
}

function stopAnswerSpeech() {
    answerSpeech.run++;   // callbacks from the previous run check this and bow out
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    resetListenButton(answerSpeech.btn);
    answerSpeech.btn = null;
}

// Chrome silently stops a single utterance after roughly 15 seconds, and
// answers run far longer, so queue the text as sentence-sized pieces.
function speechChunks(text, max = 220) {
    const chunks = [];
    let current = '';
    const push = () => { if (current.trim()) chunks.push(current.trim()); current = ''; };
    const sentences = text.replace(/\s+/g, ' ').match(/[^.!?;:]+[.!?;:]*\s*/g) || [text];
    for (const sentence of sentences) {
        if (current && (current + sentence).length > max) push();
        if (sentence.length <= max) { current += sentence; continue; }
        for (const word of sentence.split(' ')) {
            if (current && (current + ' ' + word).length > max) push();
            current += (current ? ' ' : '') + word;
        }
    }
    push();
    return chunks;
}

window.toggleSpeech = function (btn) {
    const wasPlaying = answerSpeech.btn === btn;
    stopAnswerSpeech();              // always cancel whatever is speaking first
    if (wasPlaying) return;          // second click on the same answer = stop

    if (!('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') {
        showToast('Read-aloud is not supported in this browser.');
        return;
    }
    const body = btn.closest('.bot-message')?.querySelector('.markdown-body');
    const chunks = speechChunks((body?.innerText || '').trim());
    if (!chunks.length) return;

    const run = answerSpeech.run;
    answerSpeech.btn = btn;
    btn.classList.add('speaking');
    btn.innerHTML = '<i class="ri-stop-circle-line"></i> <span>Stop</span>';
    btn.title = 'Stop reading';

    const lang = SPEECH_LANGS[state.lang] || 'en-IN';
    // Chrome can drop a speak() issued in the same tick as cancel(); wait a beat.
    setTimeout(() => {
        if (answerSpeech.run !== run) return;
        const synth = window.speechSynthesis;
        chunks.forEach((chunk, i) => {
            const utter = new SpeechSynthesisUtterance(chunk);
            utter.lang = lang;
            if (i === chunks.length - 1) {
                utter.onend = () => {
                    if (answerSpeech.run !== run) return;
                    resetListenButton(btn);
                    answerSpeech.btn = null;
                };
            }
            utter.onerror = (e) => {
                if (answerSpeech.run !== run || e.error === 'interrupted' || e.error === 'canceled') return;
                console.warn('Read-aloud failed:', e.error);
                stopAnswerSpeech();
                showToast('Could not read this answer aloud.');
            };
            synth.speak(utter);
        });
        if (synth.paused) synth.resume();
    }, 60);
};

// Chrome keeps speaking across a reload unless told otherwise.
window.addEventListener('pagehide', () => {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
});

/* ── Stats modal ────────────────────────────────────────────────────────── */

async function openStats() {
    const modal = document.getElementById('stats-modal');
    modal.classList.add('active');
    const content = document.getElementById('stats-content');
    content.innerHTML = '<div class="typing-indicator" style="justify-content:center"><span></span><span></span><span></span></div>';

    try {
        const [data, store] = await Promise.all([
            api('/stats'),
            api('/api/stats').catch(() => null),
        ]);

        let html = `
            <div class="stat-row"><span class="stat-label">Model status</span><span class="stat-val ${data.model_loaded ? 'conf-high' : 'conf-low'}">${data.model_loaded ? 'Loaded' : 'Not Loaded'}</span></div>
            <div class="stat-row"><span class="stat-label">Queries Served</span><span class="stat-val">${data.queries_served}</span></div>
            <div class="stat-row"><span class="stat-label">Uptime</span><span class="stat-val">${data.uptime_s}s</span></div>
            <div class="stat-row"><span class="stat-label">LLM GPU</span><span class="stat-val">GPU ${data.llm_gpu}</span></div>
            <div class="stat-row"><span class="stat-label">Embed GPU</span><span class="stat-val">GPU ${data.embed_gpu}</span></div>
        `;

        if (store) {
            html += `<h4 style="margin-top:20px;margin-bottom:10px;color:var(--primary)">Chat store</h4>
                <div class="stat-row"><span class="stat-label">Conversations</span><span class="stat-val">${store.conversations}</span></div>
                <div class="stat-row"><span class="stat-label">Messages (all users)</span><span class="stat-val">${store.messages}</span></div>
                <div class="stat-row"><span class="stat-label">Your messages</span><span class="stat-val">${store.your_messages}</span></div>
                <div class="stat-row"><span class="stat-label">Incomplete</span><span class="stat-val">${store.incomplete_messages}</span></div>
                <div class="stat-row"><span class="stat-label">Schema</span><span class="stat-val">v${store.schema_version}</span></div>`;
        }

        html += `<h4 style="margin-top:20px;margin-bottom:10px;color:var(--primary)">GPU Memory</h4>`;
        if (data.gpu_info && data.gpu_info.length > 0) {
            data.gpu_info.forEach(g => {
                if (g.index === data.llm_gpu || g.index === data.embed_gpu) {
                    html += `<div class="stat-row">
                        <span class="stat-label">GPU ${g.index} (${g.name})</span>
                        <span class="stat-val">${g.used_gb} / ${g.total_gb} GB</span>
                    </div>`;
                }
            });
        }
        content.innerHTML = html;
    } catch (e) {
        content.innerHTML = `<p style="color:var(--danger)">Failed to fetch stats</p>`;
    }
}

function closeStats() {
    document.getElementById('stats-modal').classList.remove('active');
}

window.openStats = openStats;
window.closeStats = closeStats;


/* Server banner */
fetch(apiUrl('/health'))
    .then(r => r.json())
    .then(d => {
        const badge = document.getElementById('server-status');
        if (!d.model_loaded) {
            badge.innerHTML = '<span class="dot"></span> Booting Model...';
            badge.style.color = 'var(--warning)';
            badge.style.borderColor = 'hsla(40, 90%, 60%, 0.2)';
            badge.style.background = 'hsla(40, 90%, 60%, 0.1)';
        }
    })
    .catch(e => console.error(e));/* ==========================================================================
   LEGAL INTELLIGENCE SUITE CLIENT ENGINE
   ========================================================================== */

state.currentSuite = 'research';
state.lang = localStorage.getItem('legalmind_lang') || 'en';
state.mootRound = 1;
state.mootHistory = [];

// Initialize Language Selector on boot
(function initSuiteState() {
    document.addEventListener('DOMContentLoaded', () => {
        const langSelect = document.getElementById('lang-select');
        if (langSelect && state.lang) {
            langSelect.value = state.lang;
        }
        onDraftTypeChange('legal_notice_138_ni');
    });
})();

/* ── 1. Suite Navigation & Workspace Switching ──────────────────────────── */

window.switchSuite = function(suiteName) {
    state.currentSuite = suiteName;

    // Auto-expand "More" if the selected feature is inside features-more-wrap
    const moreSuites = ['bail', 'firaudit', 'citcheck', 'limitation', 'pleading'];
    if (moreSuites.includes(suiteName)) {
        const wrap = document.getElementById('features-more-wrap');
        const btnText = document.getElementById('more-btn-text');
        if (wrap && (wrap.hidden || wrap.style.display === 'none')) {
            wrap.hidden = false;
            wrap.style.display = 'flex';
            if (btnText) btnText.textContent = 'Less';
        }
    }

    // Update nav tab highlights
    document.querySelectorAll('.suite-tab, .feature-item').forEach(tab => {
        if (tab.classList.contains('feature-more-btn')) return;
        if (tab.getAttribute('data-tab') === suiteName) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    // Toggle panels
    const panels = ['research', 'drafter', 'moot', 'dossier', 'temporal', 'bail', 'firaudit', 'citcheck', 'limitation', 'pleading'];
    panels.forEach(p => {
        const el = document.getElementById(`workspace-${p}`);
        if (el) {
            if (p === suiteName) {
                el.hidden = false;
                el.classList.add('active');
            } else {
                el.hidden = true;
                el.classList.remove('active');
            }
        }
    });

    // Lazy initialization on workspace activation
    if (suiteName === 'moot') {
        if (!state.mootInitialized) {
            state.mootInitialized = true;
            if (window.onBenchSelectChange) window.onBenchSelectChange();
        }
    }
};

window.toggleMoreFeatures = function() {
    const wrap = document.getElementById('features-more-wrap');
    const btnText = document.getElementById('more-btn-text');
    if (!wrap) return;
    const isHidden = wrap.hidden || wrap.style.display === 'none';
    if (isHidden) {
        wrap.hidden = false;
        wrap.style.display = 'flex';
        if (btnText) btnText.textContent = 'Less';
    } else {
        wrap.hidden = true;
        wrap.style.display = 'none';
        if (btnText) btnText.textContent = 'More';
    }
};

window.changeLanguage = function(lang) {
    state.lang = lang;
    localStorage.setItem('legalmind_lang', lang);
    console.log('Legal assistant language updated to:', lang);

    const placeholders = {
        en: 'e.g. Can bail be denied under PMLA if the twin conditions of Section 45 are not met? Cite recent SC cases.',
        hi: 'उदा. क्या पीएमएलए की धारा 45 की दोहरी शर्तें पूरी न होने पर जमानत से इनकार किया जा सकता है?',
        ta: 'எ.கா. பி.எம்.எல்.ஏ பிரிவு 45-ன் நிபந்தனைகள் பூர்த்தியாகாவிட்டால் ஜாமீன் மறுக்கப்படலாமா?'
    };
    const input = document.getElementById('query-input');
    if (input && placeholders[lang]) {
        input.placeholder = placeholders[lang];
    }
};

/* ── 2. AI Document Drafter & Contract Redlining ────────────────────────── */

window.toggleDrafterSubtab = function(mode) {
    const draftView = document.getElementById('drafter-mode-draft');
    const redlineView = document.getElementById('drafter-mode-redline');
    const draftBtn = document.getElementById('subtab-draft-btn');
    const redlineBtn = document.getElementById('subtab-redline-btn');

    if (mode === 'draft') {
        draftView.hidden = false;
        redlineView.hidden = true;
        draftBtn.classList.add('active');
        redlineBtn.classList.remove('active');
    } else {
        draftView.hidden = true;
        redlineView.hidden = false;
        draftBtn.classList.remove('active');
        redlineBtn.classList.add('active');
    }
};

window.onDraftTypeChange = function(type) {
    const container = document.getElementById('draft-dynamic-fields');
    if (!container) return;

    if (type === 'legal_notice_138_ni') {
        container.innerHTML = `
            <div class="form-group">
                <label>Complainant / Payee Name</label>
                <input type="text" id="df-sender" class="form-control" value="Alpha Infotech Private Limited">
            </div>
            <div class="form-group">
                <label>Advocate for Complainant</label>
                <input type="text" id="df-advocate" class="form-control" value="Advocate R. S. Narayanan, High Court">
            </div>
            <div class="form-group">
                <label>Drawer / Accused Name & Address</label>
                <input type="text" id="df-receiver" class="form-control" value="Zenith Engineering Solutions Ltd, Andheri East, Mumbai">
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="form-group">
                    <label>Cheque Number</label>
                    <input type="text" id="df-chequeno" class="form-control" value="738291">
                </div>
                <div class="form-group">
                    <label>Cheque Amount (INR)</label>
                    <input type="text" id="df-amount" class="form-control" value="₹ 12,50,000/-">
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="form-group">
                    <label>Drawn Bank</label>
                    <input type="text" id="df-bank" class="form-control" value="HDFC Bank, Fort Branch">
                </div>
                <div class="form-group">
                    <label>Dishonour Memo Reason</label>
                    <input type="text" id="df-reason" class="form-control" value="Funds Insufficient">
                </div>
            </div>
        `;
    } else if (type === 'bail_application') {
        container.innerHTML = `
            <div class="form-group">
                <label>Name of Accused / Petitioner</label>
                <input type="text" id="df-accused" class="form-control" value="Vikramaditya Rao">
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="form-group">
                    <label>Police Station & Crime / FIR No.</label>
                    <input type="text" id="df-fir" class="form-control" value="Cyber Crime P.S., FIR No. 142/2024">
                </div>
                <div class="form-group">
                    <label>Charged Provisions</label>
                    <input type="text" id="df-sections" class="form-control" value="S. 316, 318 BNS (S. 406, 420 IPC) & S. 66D IT Act">
                </div>
            </div>
            <div class="form-group">
                <label>Duration of Custody / Incarceration</label>
                <input type="text" id="df-custody" class="form-control" value="95 days in judicial custody; chargesheet filed">
            </div>
            <div class="form-group">
                <label>Primary Grounds for Bail</label>
                <textarea id="df-grounds" class="form-control" rows="3">Investigation complete; chargesheet filed; no flight risk; petitioner is permanent resident with deep roots in society; case based entirely on documentary evidence already seized.</textarea>
            </div>
        `;
    } else if (type === 'commercial_nda') {
        container.innerHTML = `
            <div class="form-group">
                <label>Disclosing Party</label>
                <input type="text" id="df-disclosing" class="form-control" value="Nexus Artificial Intelligence Systems Pvt Ltd">
            </div>
            <div class="form-group">
                <label>Receiving Party</label>
                <input type="text" id="df-receiving" class="form-control" value="Apex Global Cloud Solutions LLP">
            </div>
            <div class="form-group">
                <label>Purpose of Information Sharing</label>
                <input type="text" id="df-purpose" class="form-control" value="Evaluating strategic enterprise integration and joint algorithmic research">
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="form-group">
                    <label>Confidentiality Term (Years)</label>
                    <input type="text" id="df-term" class="form-control" value="3 Years from Disclosure">
                </div>
                <div class="form-group">
                    <label>Governing Law & Seat</label>
                    <input type="text" id="df-jurisdiction" class="form-control" value="Indian Law; Exclusive Jurisdiction of Courts at Bengaluru">
                </div>
            </div>
        `;
    } else {
        container.innerHTML = `
            <div class="form-group">
                <label>Complainant Name</label>
                <input type="text" id="df-comp-name" class="form-control" value="Dr. Sunita Deshmukh">
            </div>
            <div class="form-group">
                <label>Opposite Party (Company / Service Provider)</label>
                <input type="text" id="df-opposite-party" class="form-control" value="Reliable Health Insurance Corporation Ltd">
            </div>
            <div class="form-group">
                <label>Deficiency in Service & Dispute Facts</label>
                <textarea id="df-deficiency" class="form-control" rows="3">Wrongful and arbitrary repudiation of cashless medical insurance claim amounting to Rs. 4,75,000 for emergency cardiac stenting.</textarea>
            </div>
            <div class="form-group">
                <label>Compensation & Relief Claimed</label>
                <input type="text" id="df-comp-relief" class="form-control" value="Reimbursement of Rs. 4,75,000 with 12% interest + Rs. 1,00,000 for mental agony">
            </div>
        `;
    }
};

window.generateDocumentDraft = async function() {
    const typeSelect = document.getElementById('draft-type-select');
    const docType = typeSelect ? typeSelect.value : 'legal_notice_138_ni';
    const btn = document.getElementById('btn-generate-draft');
    const editor = document.getElementById('draft-output-area');
    const title = document.getElementById('draft-preview-title');

    const params = {};
    if (docType === 'legal_notice_138_ni') {
        params.sender_name = document.getElementById('df-sender')?.value || 'Alpha Infotech';
        params.advocate_name = document.getElementById('df-advocate')?.value || 'Advocate R. S. Narayanan';
        params.receiver_name = document.getElementById('df-receiver')?.value || 'Zenith Engineering';
        params.cheque_number = document.getElementById('df-chequeno')?.value || '738291';
        params.amount = document.getElementById('df-amount')?.value || '₹ 12,50,000/-';
        params.bank_name = document.getElementById('df-bank')?.value || 'HDFC Bank';
        params.memo_reason = document.getElementById('df-reason')?.value || 'Funds Insufficient';
    } else if (docType === 'bail_application') {
        params.accused_name = document.getElementById('df-accused')?.value || 'Petitioner Accused';
        params.fir_details = document.getElementById('df-fir')?.value || 'FIR No. 142/2024';
        params.sections = document.getElementById('df-sections')?.value || 'S. 316, 318 BNS';
        params.custody_period = document.getElementById('df-custody')?.value || '95 days';
        params.grounds = document.getElementById('df-grounds')?.value || 'Grounds for bail';
    } else if (docType === 'commercial_nda') {
        params.disclosing_party = document.getElementById('df-disclosing')?.value || 'Nexus Systems';
        params.receiving_party = document.getElementById('df-receiving')?.value || 'Apex Global';
        params.purpose = document.getElementById('df-purpose')?.value || 'Enterprise integration';
        params.term = document.getElementById('df-term')?.value || '3 Years';
        params.jurisdiction = document.getElementById('df-jurisdiction')?.value || 'Bengaluru';
    } else {
        params.complainant = document.getElementById('df-comp-name')?.value || 'Complainant';
        params.opposite_party = document.getElementById('df-opposite-party')?.value || 'Opposite Party';
        params.deficiency = document.getElementById('df-deficiency')?.value || 'Deficiency in service';
        params.relief = document.getElementById('df-comp-relief')?.value || 'Relief claimed';
    }

    try {
        if (btn) btn.disabled = true;
        editor.value = 'Analyzing statutory templates and generating court-ready legal draft...';
        
        const res = await api('/api/drafter/generate', {
            method: 'POST',
            body: JSON.stringify({ document_type: docType, parameters: params })
        });

        if (res && res.draft_text) {
            editor.value = res.draft_text;
            if (title) title.innerText = res.title || 'Court-Ready Document Draft';
        }
    } catch (e) {
        console.error('Draft generation error:', e);
        editor.value = `Error generating draft: ${e.message}`;
    } finally {
        if (btn) btn.disabled = false;
    }
};

window.copyDraftText = function() {
    const editor = document.getElementById('draft-output-area');
    if (editor && editor.value) {
        navigator.clipboard.writeText(editor.value).then(() => {
            alert('Draft copied to clipboard!');
        });
    }
};

window.downloadDraftText = function() {
    const editor = document.getElementById('draft-output-area');
    if (!editor || !editor.value) return;
    const blob = new Blob([editor.value], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LegalMind_Draft_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
};

window.auditCurrentDraft = function() {
    const editor = document.getElementById('draft-output-area');
    if (!editor || !editor.value) {
        alert('Please generate or enter draft text first.');
        return;
    }
    toggleDrafterSubtab('redline');
    const auditInput = document.getElementById('contract-audit-text');
    if (auditInput) {
        auditInput.value = editor.value;
        auditContractClause();
    }
};

window.loadSampleClause = function(type) {
    const auditInput = document.getElementById('contract-audit-text');
    if (!auditInput) return;

    if (type === 'indemnity') {
        auditInput.value = "The Vendor shall unconditionally defend, indemnify, and hold harmless the Company, its directors, employees, and affiliates from and against any and all losses, claims, damages, liabilities, and expenses whatsoever, without any limitation of liability or cap, arising out of any breach, negligence, or statutory non-compliance.";
    } else if (type === 'non_compete') {
        auditInput.value = "During the term of this agreement and for a period of five (5) years following termination thereof, the Employee/Consultant shall not directly or indirectly engage in, consult for, or establish any business competing with the Company anywhere within the territory of India.";
    } else if (type === 'arbitration') {
        auditInput.value = "Any dispute or difference arising out of this Agreement shall be referred to a Sole Arbitrator nominated and appointed unilaterally by the Managing Director of the Company. The decision of such arbitrator shall be final and binding upon both parties.";
    }
};

window.auditContractClause = async function() {
    const auditInput = document.getElementById('contract-audit-text');
    const btn = document.getElementById('btn-audit-contract');
    const resultsContainer = document.getElementById('redline-content');
    const placeholder = document.getElementById('redline-placeholder');

    if (!auditInput || !auditInput.value.trim()) {
        alert('Please paste a contract clause or agreement to audit.');
        return;
    }

    try {
        if (btn) btn.disabled = true;
        if (placeholder) placeholder.hidden = true;
        if (resultsContainer) {
            resultsContainer.hidden = false;
            resultsContainer.innerHTML = '<div style="padding: 30px; text-align: center;"><i class="ri-loader-4-line spin" style="font-size: 2rem; color: var(--apple-blue);"></i><p style="margin-top: 10px; color: var(--text-muted);">Auditing clause against Indian Contract Act & Supreme Court precedents...</p></div>';
        }

        const res = await api('/api/drafter/review', {
            method: 'POST',
            body: JSON.stringify({ contract_text: auditInput.value.trim() })
        });

        const riskClass = (res.risk_level || 'low').toLowerCase();
        let issuesHtml = '';
        if (res.issues && res.issues.length > 0) {
            issuesHtml = res.issues.map(iss => `
                <div class="issue-card">
                    <div class="issue-card-head">
                        <span style="color: var(--danger);"><i class="ri-alert-line"></i> ${iss.category || 'Statutory Risk'}</span>
                        <span class="risk-badge ${riskClass}">${iss.severity || 'HIGH'}</span>
                    </div>
                    <div class="issue-desc">${iss.description || ''}</div>
                    <div style="font-size: 0.78rem; color: var(--apple-azure); margin-top: 6px;">
                        <strong>Authority:</strong> ${iss.precedent_or_statute || 'Indian Contract Act, 1872'}
                    </div>
                </div>
            `).join('');
        } else {
            issuesHtml = '<p style="color: var(--success); font-size: 0.9rem; padding: 10px;"><i class="ri-checkbox-circle-line"></i> No severe statutory restrictions or unenforceability bars detected in this excerpt.</p>';
        }

        resultsContainer.innerHTML = `
            <div class="risk-summary-badge-wrap">
                <div>
                    <h4 style="font-size: 1rem; margin-bottom: 2px;">Overall Contract Risk Audit</h4>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">${res.clauses_analyzed || 1} clause(s) analyzed against Indian statutory precedents</span>
                </div>
                <div class="risk-badge ${riskClass}">
                    <i class="ri-shield-alert-line"></i> ${res.risk_level || 'EVALUATED'} RISK (${res.risk_score || 75}/100)
                </div>
            </div>

            <h4 style="font-size: 0.9rem; margin-bottom: 10px; color: var(--text-main);">Identified Vulnerabilities</h4>
            ${issuesHtml}

            <div class="redline-revision-box">
                <h4><i class="ri-sparkling-fill"></i> Recommended Enforceable Safe Revision</h4>
                <div style="font-family: var(--font-body); font-size: 0.88rem; line-height: 1.5; color: var(--text-main); white-space: pre-wrap; background: rgba(0,0,0,0.15); padding: 12px; border-radius: 8px;">${res.safe_draft || 'Revision tailored to balance party protection and Indian law enforceability.'}</div>
            </div>
        `;
    } catch (e) {
        console.error('Audit error:', e);
        if (resultsContainer) resultsContainer.innerHTML = `<p style="color: var(--danger);">Contract review failed: ${e.message}</p>`;
    } finally {
        if (btn) btn.disabled = false;
    }
};



/* ── 4. Moot Court Simulator Engine ("Judge Mode") ──────────────────────── */

// Coram metadata dictionary with 5 specialized Indian Appellate Benches
const MOOT_CORAMS = {
    constitutional: {
        title: "Constitution Bench of the Supreme Court of India",
        coram: "5-Judge Constitution Coram",
        presiding: "Hon'ble Presiding Judge (Senior Constitutional Jurist)",
        jurisdiction: "Part III Fundamental Rights, Basic Structure & Proportionality",
        avatar: '<i class="ri-scales-3-line"></i>',
        greeting: "Counsel, this Court is convened to hear your submissions on constitutional validity and fundamental rights. State your primary proposition clearly and be prepared to address the four-prong proportionality standard.",
        samples: [
            { label: "Privacy Warrant (Art. 21)", text: "Article 21 incorporates substantive due process and procedural fairness under Maneka Gandhi; executive surveillance without prior judicial warrant violates the four-prong proportionality test in Puttaswamy." },
            { label: "Manifest Arbitrariness (Art. 14)", text: "The executive notification suffers from manifest arbitrariness under Shayara Bano as it was enacted capriciously and without any determining legal principle." },
            { label: "Basic Structure (Art. 368)", text: "The constitutional amendment damages the basic structure by extinguishing judicial review of executive appointments contrary to Kesavananda Bharati." }
        ],
        statutes: [
            { code: "Art. 14", desc: "Equality before Law & Non-Arbitrariness" },
            { code: "Art. 19(1)(a)", desc: "Freedom of Speech & Expression" },
            { code: "Art. 21", desc: "Right to Life & Personal Liberty" },
            { code: "Art. 32", desc: "Constitutional Remedies & Prerogative Writs" }
        ],
        cases: [
            { name: "Maneka Gandhi v. UOI (1978) 1 SCC 248", cite: "Maneka Gandhi (1978)" },
            { name: "Justice K.S. Puttaswamy v. UOI (2017) 10 SCC 1", cite: "Puttaswamy (2017)" },
            { name: "Shayara Bano v. UOI (2017) 9 SCC 1", cite: "Shayara Bano (2017)" }
        ]
    },
    criminal: {
        title: "Criminal Appellate Division of the Supreme Court of India",
        coram: "3-Judge Criminal Appellate Coram",
        presiding: "Hon'ble Justice (Criminal Jurisprudence Specialist)",
        jurisdiction: "Appellate Bail, BNSS 2023 / CrPC 1973, PMLA Statutory Bars & Electronic Evidence",
        avatar: '<i class="ri-scales-2-line"></i>',
        greeting: "Counsel, we are dealing with serious penal allegations. State how your client crosses the statutory threshold and why custodial interrogation is not necessary on the material on record.",
        samples: [
            { label: "PMLA Bail Bar (Art. 21)", text: "Under PMLA Section 45, prolonged pre-trial incarceration without trial overrides statutory bail bars as held by the Supreme Court in Manish Sisodia (2024)." },
            { label: "Quashing FIR (BNSS 528 / CrPC 482)", text: "The FIR discloses a purely civil commercial dispute; continuation of criminal proceedings constitutes an abuse of process under Bhajan Lal Category 1 and 7." },
            { label: "Anticipatory Bail (BNSS 482 / CrPC 438)", text: "Under Satender Antil and Arnesh Kumar, custodial interrogation is unwarranted where the accused has fully cooperated with statutory summons." }
        ],
        statutes: [
            { code: "PMLA Sec 45", desc: "Twin Conditions for Bail" },
            { code: "BNSS Sec 480 / CrPC 437", desc: "Bail in Non-Bailable Offences" },
            { code: "BNSS Sec 482 / CrPC 438", desc: "Direction for Grant of Anticipatory Bail" },
            { code: "BSA 2023 Sec 63", desc: "Admissibility of Electronic Records" }
        ],
        cases: [
            { name: "Manish Sisodia v. ED (2024) INSC 595", cite: "Manish Sisodia (2024)" },
            { name: "Vijay Madanlal Choudhary v. UOI (2022) SCC OnLine SC 929", cite: "Vijay Madanlal (2022)" },
            { name: "Satender Kumar Antil v. CBI (2022) 10 SCC 51", cite: "Satender Antil (2022)" }
        ]
    },
    commercial: {
        title: "Commercial & Arbitration Appellate Division",
        coram: "Division Bench (Courtroom No. 7)",
        presiding: "Hon'ble Justice (Commercial Law Division)",
        jurisdiction: "Contractual Damages (S.73/74 ICA), Section 34/37 Arbitration Act & Specific Relief",
        avatar: '<i class="ri-briefcase-4-line"></i>',
        greeting: "Counsel, turn your attention to the text of the commercial agreement and governing arbitration clause. How do you establish legal injury under Section 73/74 of the Contract Act?",
        samples: [
            { label: "Contract Damages (S. 74)", text: "The liquidated damages clause is a genuine pre-estimate of loss under Section 74 of the Contract Act and does not require proof of actual damage under ONGC v. Saw Pipes." },
            { label: "Arbitral Award Setting Aside (S. 34)", text: "The arbitral award suffers from patent illegality appearing on the face of the award under Associate Builders as the arbitrator rewrote express contractual terms." },
            { label: "Section 37 Supervisory Scope", text: "Under Ssangyong (2019), appellate courts under Section 37 cannot sit as a court of appeal or re-appreciate evidentiary findings of the tribunal." }
        ],
        statutes: [
            { code: "ICA Sec 73", desc: "Compensation for Loss or Damage" },
            { code: "ICA Sec 74", desc: "Compensation for Breach where Penalty Stipulated" },
            { code: "A&C Act Sec 34", desc: "Application for Setting Aside Arbitral Award" },
            { code: "A&C Act Sec 37", desc: "Supervisory Scope in Arbitral Appeals" }
        ],
        cases: [
            { name: "Kailash Nath Associates v. DDA (2015) 4 SCC 136", cite: "Kailash Nath (2015)" },
            { name: "ONGC v. Saw Pipes (2003) 5 SCC 705", cite: "Saw Pipes (2003)" },
            { name: "Associate Builders v. DDA (2015) 3 SCC 49", cite: "Associate Builders (2015)" }
        ]
    },
    regulatory: {
        title: "Public Interest, Environmental & Regulatory Bench",
        coram: "Special Green & Administrative Bench",
        presiding: "Hon'ble Presiding Justice (Environmental & Public Law)",
        jurisdiction: "Environmental Jurisprudence, Natural Justice, Ultra Vires & Public Trust Doctrine",
        avatar: '<i class="ri-leaf-line"></i>',
        greeting: "Counsel, this Court exercises constitutional jurisdiction under Article 32/226 to protect public trust and environmental sustainability. Address how the impugned project complies with the Precautionary Principle.",
        samples: [
            { label: "Precautionary Principle", text: "Under Vellore Citizens and Article 21, the lack of full scientific certainty cannot justify postponing cost-effective measures to prevent environmental degradation." },
            { label: "Absolute Liability", text: "Under M.C. Mehta Oleum Gas Leak, an enterprise engaged in a hazardous activity owes an absolute, non-delegable duty to the community without Rylands v. Fletcher exceptions." },
            { label: "Environmental Rule of Law", text: "Under Hanuman Laxman Aroskar (2019), complete and transparent disclosure of baseline ecological data is an inviolable condition precedent." }
        ],
        statutes: [
            { code: "Art. 48A & 51A(g)", desc: "Protection of Environment & Forests" },
            { code: "NGT Act Sec 20", desc: "Precautionary Principle & Polluter Pays" },
            { code: "EPA 1986 Sec 3", desc: "Measures to Protect and Improve Environment" }
        ],
        cases: [
            { name: "Vellore Citizens' Welfare Forum v. UOI (1996) 5 SCC 647", cite: "Vellore Citizens (1996)" },
            { name: "M.C. Mehta v. UOI (1987) 1 SCC 395", cite: "M.C. Mehta (1987)" },
            { name: "Hanuman Laxman Aroskar v. UOI (2019) 15 SCC 401", cite: "Aroskar (2019)" }
        ]
    },
    tax_insolvency: {
        title: "Special Tax, Corporate & Insolvency Appellate Bench",
        coram: "Special Tax & Insolvency Bench (Courtroom No. 9)",
        presiding: "Hon'ble Justice (Corporate & Insolvency Jurisprudence)",
        jurisdiction: "Insolvency and Bankruptcy Code (IBC 2016), S.14 Moratorium, S.53 Waterfall & CGST Appeals",
        avatar: '<i class="ri-bar-chart-2-line"></i>',
        greeting: "Counsel, this Bench is convened to hear matters under corporate insolvency and revenue jurisprudence. How does your proposition reconcile with the non-obstante clause in Section 238 IBC and the commercial wisdom doctrine?",
        samples: [
            { label: "IBC S.53 Waterfall", text: "Under Section 238 IBC and the landmark ruling in Essar Steel (2020), the Section 53 waterfall strictly prioritizes secured financial creditors over Crown tax dues." },
            { label: "Moratorium Freeze (S. 14)", text: "Under Section 14 IBC, commencement of CIRP imposes an absolute statutory moratorium prohibiting tax authorities from attaching assets of the corporate debtor." },
            { label: "Rainbow Papers Limited", text: "As clarified in Paschimanchal Vidyut (2023), statutory first charges under State VAT enactments cannot bypass Section 53 of the Insolvency Code." }
        ],
        statutes: [
            { code: "IBC Sec 14", desc: "Moratorium on Continuation of Proceedings" },
            { code: "IBC Sec 53", desc: "Distribution of Assets (Waterfall Priority)" },
            { code: "IBC Sec 238", desc: "Overriding Effect of Code (Non-Obstante)" },
            { code: "IT Act Sec 148", desc: "Issue of Notice for Escaped Assessment" }
        ],
        cases: [
            { name: "CoC of Essar Steel v. Satish Gupta (2020) 8 SCC 531", cite: "Essar Steel (2020)" },
            { name: "Swiss Ribbons v. Union of India (2019) 4 SCC 17", cite: "Swiss Ribbons (2019)" },
            { name: "Paschimanchal Vidyut v. Raman Ispat (2023) INSC 628", cite: "Paschimanchal (2023)" }
        ]
    }
};

// Curated Moot Problems Catalog
const MOOT_PROBLEMS_CATALOG = {
    constitutional_surveillance: {
        id: "constitutional_surveillance",
        coram: "constitutional",
        title: "People's Union for Digital Rights v. Union of India",
        matrix: "The Union Government promulgated the National Digital Security & Facial Recognition Rules, 2025 under Section 69 of the IT Act. The Rules mandate real-time automated biometric surveillance across all railway transit hubs and airport check-ins, backed by predictive AI crime-risk profiling without prior judicial warrant. The Petitioner challenges the Rules as violative of Articles 14, 19(1)(a), and 21.",
        issues: [
            "Whether executive authorization without prior judicial warrant satisfies the third (least intrusive) prong of Puttaswamy proportionality?",
            "Whether algorithmic black-box risk scoring infringes Article 14 by introducing unguided discretion and manifest arbitrariness?",
            "Whether national security and transit crime prevention qualify as compelling state interests overriding individual digital privacy under Article 21?"
        ],
        petitionerSample: "May it please your Lordships, under Article 21 and the nine-judge Constitution Bench ruling in Puttaswamy, privacy is an inviolable fundamental right. The impugned executive surveillance scheme fails the four-prong proportionality standard because automated algorithmic tracking without prior judicial warrant is intrinsically disproportionate.",
        respondentSample: "May it please your Lordships, the Union Government enacted the 2025 Rules under Section 69 of the IT Act to combat urgent counter-terrorism and cross-border transit threats. Under the PUCL (1997) framework, executive oversight via high-level review committees provides constitutionally sufficient procedural safeguards."
    },
    criminal_pmla_bail: {
        id: "criminal_pmla_bail",
        coram: "criminal",
        title: "Vikramaditya Sharma v. Directorate of Enforcement",
        matrix: "The Appellant, a former non-executive director of an infrastructure consortium, is arraigned under Sections 3 and 4 of PMLA 2002 in connection with a Rs. 1,400 Crore credit facility. The Appellant has been incarcerated for 28 months in pre-trial custody with 84,000 pages of digital records and 112 listed witnesses. Charges have not yet been framed.",
        issues: [
            "Whether prolonged pre-trial incarceration where trial cannot conclude in reasonable time dilutes or overrides Section 45 PMLA twin conditions under Article 21?",
            "Whether a non-executive director without financial signing authority can be imputed with mens rea under Section 3 PMLA?",
            "How to harmonize the strict statutory threshold in Vijay Madanlal Choudhary (2022) with the liberty jurisprudence in Manish Sisodia (2024)?"
        ],
        petitionerSample: "May it please this Hon'ble Court, under Article 21 and the binding declaration in Manish Sisodia (2024), statutory bail bars under Section 45 PMLA cannot supersede the constitutional right to speedy trial when the petitioner has suffered 28 months of pre-trial incarceration without even framing of charges.",
        respondentSample: "May it please this Court, the accusations involve laundering over Rs. 1,400 Crores. Under Vijay Madanlal Choudhary (2022), Section 45 twin conditions are mandatory. The delay is attributable to the defence inspecting voluminous electronic evidence, and release on bail creates acute risks of witness tampering."
    },
    commercial_liquidated_damages: {
        id: "commercial_liquidated_damages",
        coram: "commercial",
        title: "AeroTech Infra Pvt Ltd v. National Logistics Corridor Authority",
        matrix: "AeroTech was awarded an EPC contract for constructing automated cargo transfer terminals. Delay of 14 months occurred due to global supply disruptions. NLCA terminated the contract and invoked Clause 42.1, forfeiting the entire 10% Performance Bank Guarantee (Rs. 82 Crores) as liquidated damages without adducing evidence of actual loss. The Commercial Division set aside the arbitral award in favour of AeroTech under Section 34.",
        issues: [
            "Whether under Section 74 of the Contract Act, liquidated damages can be forfeited without proving actual damage when the facility was completed by alternate contractors at lower costs?",
            "Whether the High Court under Section 34 impermissibly sat as a court of appeal and re-appreciated contractual terms contrary to Associate Builders and Ssangyong?",
            "Does the doctrine in Kailash Nath Associates (2015) require actual loss proof in public infrastructural agreements?"
        ],
        petitionerSample: "May it please your Lordships, the Commercial Division gravely erred in setting aside the arbitral award under Section 34. Under Ssangyong and Associate Builders, an arbitral tribunal is the ultimate master of contractual interpretation. Furthermore, under Kailash Nath (2015), liquidated damages under Section 74 cannot be forfeited without proof of actual legal injury.",
        respondentSample: "May it please your Lordships, in complex national public logistics corridors, delay causes irreparable economic and societal disruption that cannot be quantified in rupees. Under ONGC v. Saw Pipes (2003), the pre-estimated liquidated sum agreed between sophisticated commercial entities must be enforced."
    },
    regulatory_green_clearance: {
        id: "regulatory_green_clearance",
        coram: "regulatory",
        title: "Himalayan Ecological Foundation v. State Infrastructure Corp & MoEFCC",
        matrix: "The MoEFCC granted Environmental Clearance (EC) for widening a 125 km Himalayan highway passing through a fragile eco-sensitive buffer zone prone to catastrophic glacial flash floods. The EIA report omitted baseline geological seismic fault-line data and dispensed with public hearings, claiming statutory exemption for strategic connectivity.",
        issues: [
            "Whether suppression of seismic fault-line vulnerability voids the Environmental Clearance ab initio under Hanuman Laxman Aroskar (2019)?",
            "How does the Precautionary Principle apply when geological experts caution that slope excavation creates irreversible landslide risks?",
            "Can strategic national connectivity completely displace statutory requirements of public consultation under the Environment Protection Act 1986?"
        ],
        petitionerSample: "May it please this Hon'ble Bench, under Article 21 and the Precautionary Principle laid down in Vellore Citizens and Hanuman Laxman Aroskar, complete disclosure of seismic vulnerability is a non-negotiable condition precedent for environmental clearance. Omission of vital fault-line data vitiates the clearance ab initio.",
        respondentSample: "May it please your Lordships, the highway is essential for national strategic border logistics. Specialized engineering slope-stabilization mitigations have been approved by the expert appraisal committee. In technical environmental evaluations, courts must accord deference to expert regulatory bodies."
    },
    tax_ibc_priority: {
        id: "tax_ibc_priority",
        coram: "tax_insolvency",
        title: "State Tax Department v. Resolution Professional of Apex Steel Ltd & CoC",
        matrix: "Apex Steel Ltd was admitted into CIRP. The State Tax Department filed a claim of Rs. 340 Crores for unpaid VAT, asserting statutory first charge on assets under Section 48 State VAT Act. The CoC approved a Resolution Plan allocating 1% to operational government dues and 65% to secured financial creditors under Section 53 IBC. State Tax Dept challenges the plan relying on Rainbow Papers (2022).",
        issues: [
            "Whether statutory tax dues under State revenue enactments constitute 'secured creditors' ranking pari passu with commercial banks under Section 53(1)(b) IBC?",
            "How does Section 238 IBC overriding clause interact with State legislation creating statutory first charges?",
            "Does the subsequent Supreme Court ruling in Paschimanchal Vidyut (2023) confine Rainbow Papers to its peculiar statutory wording?"
        ],
        petitionerSample: "May it please your Lordships, under Section 238 of the Insolvency and Bankruptcy Code and the three-judge Bench ruling in Essar Steel, the Section 53 waterfall explicitly subordinates government dues to financial creditors. The State Tax Department's reliance on Rainbow Papers stands clarified and confined by Paschimanchal Vidyut (2023).",
        respondentSample: "May it please the Bench, Section 48 of the State VAT Act creates an express statutory charge on the property of the dealer. Under State Tax Officer v. Rainbow Papers (2022), such statutory charge constitutes the State as a secured creditor entitled to pari passu distribution under Section 53(1)(b)."
    }
};

// Argument Timer State
let mootTimerInterval = null;
let mootTimerSeconds = 300; // 5 minutes default
let mootTimerIsRunning = false;

// MediaRecorder State for Voice Pleading
let mootMediaRecorder = null;
let mootAudioChunks = [];
let mootIsRecording = false;
let activeTTSAudio = null;

// Reset Scorecard to Clean Standby State
function resetScorecardToStandby() {
    const scoreNum = document.getElementById('moot-overall-score');
    const strokeEl = document.getElementById('score-radial-stroke');
    const ratingEl = document.getElementById('moot-readiness-rating');
    const subtextEl = document.getElementById('moot-rating-subtext');
    const pillEl = document.getElementById('scorecard-status-pill');

    if (scoreNum) scoreNum.innerText = '--';
    if (strokeEl) {
        strokeEl.style.strokeDashoffset = '264';
        strokeEl.style.stroke = 'rgba(255, 255, 255, 0.15)';
    }
    if (pillEl) pillEl.innerHTML = '<span class="pulse-dot"></span> Standby';
    if (ratingEl) {
        ratingEl.innerText = 'Standby (Awaiting Opening)';
        ratingEl.style.borderColor = 'var(--border-glass)';
        ratingEl.style.color = 'var(--text-muted)';
    }
    if (subtextEl) subtextEl.innerText = 'Present your opening proposition to activate real-time advocacy analytics';

    const setMetricEmpty = (valId, barId) => {
        const vEl = document.getElementById(valId);
        const bEl = document.getElementById(barId);
        if (vEl) vEl.innerText = '--%';
        if (bEl) bEl.style.width = '0%';
    };
    setMetricEmpty('val-grounding', 'bar-grounding');
    setMetricEmpty('val-statutory', 'bar-statutory');
    setMetricEmpty('val-precedent', 'bar-precedent');
    setMetricEmpty('val-persuasion', 'bar-persuasion');

    const strList = document.getElementById('moot-strengths-list');
    if (strList) strList.innerHTML = '<li>Submissions will be evaluated on constitutional grounding, statutory precision, and courtroom poise.</li>';

    const vulList = document.getElementById('moot-vulnerabilities-list');
    if (vulList) vulList.innerHTML = '<li>Potential vulnerabilities and procedural bars will be flagged here as arguments proceed.</li>';

    const rebEl = document.getElementById('moot-rebuttal-tip');
    if (rebEl) rebEl.innerText = 'Judicial cross-examination analysis and strategic rebuttal tips will appear after Counsel addresses the Bench.';

    const cpWrap = document.getElementById('moot-counter-precedent-wrap');
    if (cpWrap) cpWrap.style.display = 'none';
}

// 1. Bench Selector Change
window.onBenchSelectChange = function(resetProblem = true) {
    const select = document.getElementById('moot-bench-select');
    const bType = select ? select.value : 'constitutional';
    const c = MOOT_CORAMS[bType] || MOOT_CORAMS.constitutional;

    const title = document.getElementById('bench-presiding-title');
    const coramBadge = document.getElementById('bench-coram-badge');
    const name = document.getElementById('bench-presiding-name');
    const juris = document.getElementById('bench-jurisdiction-text');
    const avatar = document.getElementById('bench-avatar-icon');

    if (title) title.innerText = c.title;
    if (coramBadge) coramBadge.innerText = c.coram;
    if (name) name.innerText = c.presiding;
    if (juris) juris.innerText = c.jurisdiction;
    if (avatar) avatar.innerHTML = c.avatar;

    // If resetProblem requested, clear problem banner
    if (resetProblem) {
        const probSelect = document.getElementById('moot-problem-select');
        if (probSelect) probSelect.value = 'open';
        const banner = document.getElementById('moot-problem-banner');
        if (banner) banner.style.display = 'none';

        // Update Quick Samples
        const chipsContainer = document.getElementById('moot-quick-chips');
        if (chipsContainer && c.samples) {
            chipsContainer.innerHTML = c.samples.map(s => 
                `<button type="button" class="chip-sm" onclick="fillMootArg('${s.text.replace(/'/g, "\\'")}')">${s.label}</button>`
            ).join('');
        }
    }

    // Update Compendium
    updateMootCompendiumUI(c);

    // Reset Dialogue with greeting and clean standby scorecard
    resetMootDialogueFeed(c.greeting, c.presiding);
    resetScorecardToStandby();
};

// Moot Problem Selector Change
window.onProblemSelectChange = function() {
    const probSelect = document.getElementById('moot-problem-select');
    const probKey = probSelect ? probSelect.value : 'open';
    const banner = document.getElementById('moot-problem-banner');

    if (!probKey || probKey === 'open') {
        if (banner) banner.style.display = 'none';
        onBenchSelectChange(false);
        return;
    }

    const prob = MOOT_PROBLEMS_CATALOG[probKey];
    if (!prob) return;

    // Switch Coram to match problem
    const benchSelect = document.getElementById('moot-bench-select');
    if (benchSelect && benchSelect.value !== prob.coram) {
        benchSelect.value = prob.coram;
        onBenchSelectChange(false);
    }

    // Populate Problem Banner
    if (banner) {
        banner.style.display = 'block';
        const titleEl = document.getElementById('moot-banner-case-title');
        const matrixEl = document.getElementById('moot-banner-matrix');
        const issuesEl = document.getElementById('moot-banner-issues');
        const sideTag = document.getElementById('moot-banner-side-tag');
        const sideSelect = document.getElementById('moot-side-select');
        const sideVal = sideSelect ? sideSelect.value : 'petitioner';

        if (titleEl) titleEl.innerText = prob.title;
        if (matrixEl) matrixEl.innerText = prob.matrix;
        if (issuesEl) {
            issuesEl.innerHTML = prob.issues.map(iss => `<li>${iss}</li>`).join('');
        }
        if (sideTag) {
            sideTag.innerText = sideVal === 'petitioner' ? 'Petitioner / Appellant' : 'Respondent / State';
        }
    }

    // Update Sample Quick Chips to problem-specific argument
    const chipsContainer = document.getElementById('moot-quick-chips');
    const sideSelect = document.getElementById('moot-side-select');
    const sideVal = sideSelect ? sideSelect.value : 'petitioner';
    const sampleArg = sideVal === 'petitioner' ? prob.petitionerSample : prob.respondentSample;

    if (chipsContainer) {
        chipsContainer.innerHTML = `
            <button type="button" class="chip-sm" onclick="fillMootArg('${sampleArg.replace(/'/g, "\\'")}')">
                <i class="ri-play-line"></i> Load Suggested ${sideVal === 'petitioner' ? 'Petitioner' : 'Respondent'} Opening
            </button>
            <button type="button" class="chip-sm" onclick="fillMootArg('May it please your Lordships, on Issue 1: ')">Issue 1 Submission</button>
            <button type="button" class="chip-sm" onclick="fillMootArg('May it please your Lordships, on Issue 2: ')">Issue 2 Submission</button>
        `;
    }

    // Reset dialogue feed with case title
    const c = MOOT_CORAMS[prob.coram] || MOOT_CORAMS.constitutional;
    resetMootDialogueFeed(`This Court is convened in ${prob.title}. Counsel, state your primary submissions on the framed issues.`, c.presiding);
    resetScorecardToStandby();
    showToast(`Loaded Moot Problem: ${prob.title}`, 'info');
};

// Side Selector Change
window.onSideSelectChange = function() {
    const sideSelect = document.getElementById('moot-side-select');
    const sideVal = sideSelect ? sideSelect.value : 'petitioner';
    const sideTag = document.getElementById('moot-banner-side-tag');
    if (sideTag) {
        sideTag.innerText = sideVal === 'petitioner' ? 'Petitioner / Appellant' : 'Respondent / State';
    }

    // If moot problem is active, update suggested argument
    const probSelect = document.getElementById('moot-problem-select');
    const probKey = probSelect ? probSelect.value : 'open';
    if (probKey && probKey !== 'open' && MOOT_PROBLEMS_CATALOG[probKey]) {
        const prob = MOOT_PROBLEMS_CATALOG[probKey];
        const chipsContainer = document.getElementById('moot-quick-chips');
        const sampleArg = sideVal === 'petitioner' ? prob.petitionerSample : prob.respondentSample;
        if (chipsContainer) {
            chipsContainer.innerHTML = `
                <button type="button" class="chip-sm" onclick="fillMootArg('${sampleArg.replace(/'/g, "\\'")}')">
                    <i class="ri-play-line"></i> Load Suggested ${sideVal === 'petitioner' ? 'Petitioner' : 'Respondent'} Opening
                </button>
                <button type="button" class="chip-sm" onclick="fillMootArg('May it please your Lordships, on Issue 1: ')">Issue 1 Submission</button>
                <button type="button" class="chip-sm" onclick="fillMootArg('May it please your Lordships, on Issue 2: ')">Issue 2 Submission</button>
            `;
        }
    }
    showToast(`Appearing as Counsel for the ${sideVal.toUpperCase()}.`, 'info');
};

// Toggle Problem Banner Accordion
window.toggleProblemBanner = function() {
    const body = document.getElementById('moot-banner-body');
    const arrow = document.getElementById('problem-banner-arrow');
    if (!body) return;
    if (body.style.display === 'none' || !body.style.display) {
        body.style.display = 'block';
        if (arrow) arrow.className = 'ri-arrow-up-s-line';
    } else {
        body.style.display = 'none';
        if (arrow) arrow.className = 'ri-arrow-down-s-line';
    }
};

window.onTemperamentChange = function() {
    const tempSelect = document.getElementById('moot-temperament-select');
    const val = tempSelect ? tempSelect.value : 'inquisitive';
    showToast(`Judicial temperament set to ${val.toUpperCase()} mode.`, 'info');
};

function updateMootCompendiumUI(coramObj) {
    const statDiv = document.getElementById('moot-statutory-compendium');
    const landDiv = document.getElementById('moot-landmark-compendium');

    if (statDiv && coramObj.statutes) {
        statDiv.innerHTML = coramObj.statutes.map(st => `
            <div class="compendium-chip" onclick="insertAuthority('${st.code}')" title="Insert in submission">
                <strong>${st.code}</strong> ${st.desc} <i class="ri-add-line"></i>
            </div>
        `).join('');
    }

    if (landDiv && coramObj.cases) {
        landDiv.innerHTML = coramObj.cases.map(cs => `
            <div class="compendium-chip" onclick="insertAuthority('${cs.name}')" title="Insert in submission">
                <em>${cs.cite}</em> <i class="ri-add-line"></i>
            </div>
        `).join('');
    }
}

function resetMootDialogueFeed(greetingText, presidingLabel) {
    const feed = document.getElementById('moot-dialogue-feed');
    if (!feed) return;
    feed.innerHTML = `
        <div class="moot-message judge-msg">
            <div class="moot-avatar"><i class="ri-government-line"></i></div>
            <div class="moot-bubble">
                <div class="moot-speaker-row">
                    <span class="moot-speaker">THE BENCH (${(presidingLabel || "PRESIDING JURIST").toUpperCase()}):</span>
                    <span class="moot-msg-time">Preliminary Hearing</span>
                </div>
                <p id="moot-initial-prompt">${greetingText}</p>
                <div class="moot-actions">
                    <button type="button" class="icon-pill-btn tts-btn" onclick="speakMootMessage(this)">
                        <i class="ri-volume-up-line"></i> <span>Hear the Bench</span>
                    </button>
                    <button type="button" class="icon-pill-btn-ghost" onclick="copyBenchPrompt(this)" title="Copy question">
                        <i class="ri-file-copy-line"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
    state.mootRound = 1;
    state.mootHistory = [];
    const rBadge = document.getElementById('moot-round-badge');
    if (rBadge) rBadge.innerText = 'Round 1 of 5';
}

// 2. Input Handling
window.fillMootArg = function(text) {
    const input = document.getElementById('moot-arg-input');
    if (input) {
        input.value = text;
        input.focus();
        updateMootWordCount();
    }
};

window.insertAuthority = function(authText) {
    const input = document.getElementById('moot-arg-input');
    if (!input) return;
    const cur = input.value.trim();
    if (cur) {
        input.value = `${cur} [Relying on ${authText}]`;
    } else {
        input.value = `May it please your Lordships, under ${authText}, `;
    }
    input.focus();
    updateMootWordCount();
    showToast(`Inserted "${authText}" into argument.`, 'info');
};

window.handleMootTextareaKey = function(e) {
    updateMootWordCount();
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        submitMootArgument();
    }
};

function updateMootWordCount() {
    const input = document.getElementById('moot-arg-input');
    const countEl = document.getElementById('moot-char-count');
    if (!input || !countEl) return;
    const words = input.value.trim() ? input.value.trim().split(/\s+/).length : 0;
    countEl.innerText = `${words} words`;
}

// 3. Argument Timer
window.toggleMootTimer = function() {
    const btn = document.getElementById('btn-timer-toggle');
    const icon = document.getElementById('timer-toggle-icon');
    const box = document.getElementById('moot-timer-box');

    if (mootTimerIsRunning) {
        clearInterval(mootTimerInterval);
        mootTimerIsRunning = false;
        if (icon) icon.className = 'ri-play-fill';
        if (box) box.classList.remove('timer-running');
    } else {
        mootTimerIsRunning = true;
        if (icon) icon.className = 'ri-pause-fill';
        if (box) box.classList.add('timer-running');
        mootTimerInterval = setInterval(() => {
            if (mootTimerSeconds > 0) {
                mootTimerSeconds--;
                renderMootTimerDisplay();
                if (mootTimerSeconds <= 60 && box) {
                    box.classList.add('timer-warning');
                }
            } else {
                clearInterval(mootTimerInterval);
                mootTimerIsRunning = false;
                if (icon) icon.className = 'ri-play-fill';
                showToast("Counsel's allocated argument time has expired. Conclude submissions.", 'warning');
            }
        }, 1000);
    }
};

window.resetMootTimer = function() {
    clearInterval(mootTimerInterval);
    mootTimerIsRunning = false;
    mootTimerSeconds = 300;
    const icon = document.getElementById('timer-toggle-icon');
    if (icon) icon.className = 'ri-play-fill';
    const box = document.getElementById('moot-timer-box');
    if (box) {
        box.classList.remove('timer-running', 'timer-warning');
    }
    renderMootTimerDisplay();
};

function renderMootTimerDisplay() {
    const disp = document.getElementById('moot-timer-display');
    if (!disp) return;
    const mins = Math.floor(mootTimerSeconds / 60);
    const secs = mootTimerSeconds % 60;
    disp.innerText = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

// 4. Voice Pleading (Microphone Dictation)
window.toggleMootVoicePleading = async function() {
    const btn = document.getElementById('btn-voice-moot');
    const icon = document.getElementById('voice-moot-icon');
    const label = document.getElementById('voice-moot-label');
    const input = document.getElementById('moot-arg-input');

    if (mootIsRecording) {
        // Stop recording
        if (mootMediaRecorder && mootMediaRecorder.state !== 'inactive') {
            mootMediaRecorder.stop();
        }
        mootIsRecording = false;
        if (btn) btn.classList.remove('recording');
        if (icon) icon.className = 'ri-mic-line';
        if (label) label.innerText = 'Transcribing...';
        return;
    }

    // Try SpeechRecognition first if available
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
        try {
            const recognition = new SpeechRec();
            recognition.lang = state.lang || 'en-IN';
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;

            if (btn) btn.classList.add('recording');
            if (icon) icon.className = 'ri-mic-fill';
            if (label) label.innerText = 'Listening...';
            mootIsRecording = true;

            recognition.onresult = (event) => {
                const speechResult = event.results[0][0].transcript;
                if (input) {
                    input.value = input.value.trim() ? `${input.value} ${speechResult}` : speechResult;
                    input.focus();
                    updateMootWordCount();
                }
            };
            recognition.onerror = () => {
                showToast('Voice dictation stopped or no speech detected.', 'info');
            };
            recognition.onend = () => {
                mootIsRecording = false;
                if (btn) btn.classList.remove('recording');
                if (icon) icon.className = 'ri-mic-line';
                if (label) label.innerText = 'Voice Pleading';
            };
            recognition.start();
            return;
        } catch (e) {
            console.warn('Web Speech Recognition fallback to MediaRecorder:', e);
        }
    }

    // Fallback to MediaRecorder + /api/transcribe
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mootAudioChunks = [];
        mootMediaRecorder = new MediaRecorder(stream);
        mootMediaRecorder.ondataavailable = e => { if (e.data.size > 0) mootAudioChunks.push(e.data); };
        mootMediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(mootAudioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('file', blob, 'submission.webm');
            formData.append('language', state.lang || 'en');

            try {
                const res = await fetch(apiUrl('/api/transcribe'), { method: 'POST', body: formData });
                const data = await res.json();
                if (data.text && input) {
                    input.value = input.value.trim() ? `${input.value} ${data.text}` : data.text;
                    input.focus();
                    updateMootWordCount();
                    showToast('Submission transcribed successfully.', 'success');
                }
            } catch (err) {
                console.error('Audio transcription error:', err);
                showToast('Voice transcription failed.', 'error');
            } finally {
                if (label) label.innerText = 'Voice Pleading';
            }
        };

        mootMediaRecorder.start();
        mootIsRecording = true;
        if (btn) btn.classList.add('recording');
        if (icon) icon.className = 'ri-mic-fill';
        if (label) label.innerText = 'Recording...';
    } catch (err) {
        console.error('Microphone access denied:', err);
        showToast('Microphone access required for Voice Pleading.', 'error');
    }
};

// 5. Submit Oral Argument to Judicial Bench
window.submitMootArgument = async function() {
    const input = document.getElementById('moot-arg-input');
    const feed = document.getElementById('moot-dialogue-feed');
    const btn = document.getElementById('btn-submit-moot');
    const benchSelect = document.getElementById('moot-bench-select');
    const tempSelect = document.getElementById('moot-temperament-select');

    if (!input || !input.value.trim()) {
        showToast('Counsel must submit an oral argument or legal proposition.');
        return;
    }
    const argumentText = input.value.trim();
    input.value = '';
    updateMootWordCount();

    const nowStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Append Counsel's oral submission
    const advMsg = document.createElement('div');
    advMsg.className = 'moot-message advocate-msg';
    advMsg.innerHTML = `
        <div class="moot-avatar"><i class="ri-user-voice-line"></i></div>
        <div class="moot-bubble">
            <div class="moot-speaker-row">
                <span class="moot-speaker">COUNSEL ORAL SUBMISSION:</span>
                <span class="moot-msg-time">Round ${state.mootRound || 1} • ${nowStr}</span>
            </div>
            <p>${argumentText}</p>
        </div>
    `;
    feed.appendChild(advMsg);
    feed.scrollTop = feed.scrollHeight;

    // Loading indicator for Judge with active seconds counter
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'moot-message judge-msg';
    const timerStart = Date.now();
    loadingMsg.innerHTML = `
        <div class="moot-avatar"><i class="ri-government-line"></i></div>
        <div class="moot-bubble">
            <div class="moot-speaker-row">
                <span class="moot-speaker">THE BENCH:</span>
                <span class="moot-msg-time moot-timer-tag">Deliberating (0s)...</span>
            </div>
            <p><i class="ri-loader-4-line spin"></i> The Hon'ble Bench is scrutinizing your proposition against binding Constitution Bench authorities...</p>
        </div>
    `;
    feed.appendChild(loadingMsg);
    feed.scrollTop = feed.scrollHeight;

    const timerTag = loadingMsg.querySelector('.moot-timer-tag');
    const timerInterval = setInterval(() => {
        if (timerTag) {
            const elapsed = Math.floor((Date.now() - timerStart) / 1000);
            timerTag.textContent = `Deliberating (${elapsed}s)...`;
        }
    }, 1000);

    // Deliberation timeout guard
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 75000);

    try {
        if (btn) btn.disabled = true;
        const bType = benchSelect ? benchSelect.value : 'constitutional';
        const temperament = tempSelect ? tempSelect.value : 'inquisitive';
        const sideSelect = document.getElementById('moot-side-select');
        const counselSide = sideSelect ? sideSelect.value : 'petitioner';
        const probSelect = document.getElementById('moot-problem-select');
        const probKey = probSelect ? probSelect.value : 'open';
        const prob = (probKey && probKey !== 'open') ? MOOT_PROBLEMS_CATALOG[probKey] : null;

        const res = await api('/api/moot/interject', {
            method: 'POST',
            signal: controller.signal,
            body: JSON.stringify({
                argument: argumentText,
                bench_type: bType,
                temperament: temperament,
                round_num: state.mootRound || 1,
                counsel_side: counselSide,
                case_topic: prob ? prob.title : null,
                factual_matrix: prob ? prob.matrix : null,
                history: state.mootHistory || []
            })
        });
        clearInterval(timerInterval);
        clearTimeout(timeout);

        // The service reports some refusals as 200 + { success: false, error }.
        if (!res || res.success === false) {
            throw new Error((res && res.error) || 'The Bench returned no response.');
        }

        // Remove loading
        loadingMsg.remove();

        // Render Judge's interjection
        const judgeMsg = document.createElement('div');
        judgeMsg.className = 'moot-message judge-msg';
        const judgeTitle = res.presiding_judge || "Hon'ble Presiding Judge";
        judgeMsg.innerHTML = `
            <div class="moot-avatar"><i class="ri-government-line"></i></div>
            <div class="moot-bubble">
                <div class="moot-speaker-row">
                    <span class="moot-speaker">THE BENCH (${judgeTitle.toUpperCase()}):</span>
                    <span class="moot-msg-time">Round ${state.mootRound || 1} • ${nowStr}</span>
                </div>
                <p>${res.judicial_interjection || 'Counsel, what is your binding statutory foundation?'}</p>
                <div class="moot-actions">
                    <button type="button" class="icon-pill-btn tts-btn" onclick="speakMootMessage(this)">
                        <i class="ri-volume-up-line"></i> <span>Hear the Bench</span>
                    </button>
                    <button type="button" class="icon-pill-btn-ghost" onclick="copyBenchPrompt(this)" title="Copy question">
                        <i class="ri-file-copy-line"></i>
                    </button>
                </div>
            </div>
        `;
        feed.appendChild(judgeMsg);
        feed.scrollTop = feed.scrollHeight;

        // Update Scorecard & Radar
        updateAdvocacyScorecardUI(res);

        // Update session state
        state.mootRound = (state.mootRound || 1) + 1;
        if (!state.mootHistory) state.mootHistory = [];
        state.mootHistory.push({ counsel: argumentText, bench: res.judicial_interjection });

        const rBadge = document.getElementById('moot-round-badge');
        if (rBadge) rBadge.innerText = `Round ${state.mootRound} of 5`;

    } catch (e) {
        console.error('Moot court simulation error:', e);
        const reason = e && e.name === 'AbortError'
            ? 'The Bench did not respond within 30 seconds. Please try again.'
            : (e && e.message) || 'Unknown error.';
        // Resolve the loading bubble into a readable error, keeping its frame
        const time = loadingMsg.querySelector('.moot-msg-time');
        if (time) time.textContent = 'Could not respond';
        const body = loadingMsg.querySelector('.moot-bubble p');
        if (body) body.outerHTML = `<p style="color: var(--danger);">Bench simulation error: ${escapeHtml(reason)}</p>`;
        feed.scrollTop = feed.scrollHeight;
        // Give counsel their submission back rather than losing it
        if (!input.value.trim()) { input.value = argumentText; updateMootWordCount(); }
    } finally {
        clearTimeout(timeout);
        if (btn) btn.disabled = false;
    }
};

// 6. Update Scorecard UI & SVG Radial Gauge
function updateAdvocacyScorecardUI(data) {
    const sc = data.scorecard || {};
    const overall = Math.round(sc.overall_score || 78);

    const pillEl = document.getElementById('scorecard-status-pill');
    if (pillEl) pillEl.innerHTML = '<span class="pulse-dot"></span> Live Evaluation';

    // SVG Radial Progress calculation (circumference = 2 * PI * 42 ~= 264)
    const strokeEl = document.getElementById('score-radial-stroke');
    const scoreNumEl = document.getElementById('moot-overall-score');
    const ratingEl = document.getElementById('moot-readiness-rating');
    const subtextEl = document.getElementById('moot-rating-subtext');

    if (scoreNumEl) scoreNumEl.innerText = overall;
    if (strokeEl) {
        const offset = Math.max(0, Math.min(264, 264 - (264 * (overall / 100))));
        strokeEl.style.strokeDashoffset = offset;
        if (overall >= 85) strokeEl.style.stroke = 'var(--success)';
        else if (overall >= 74) strokeEl.style.stroke = 'var(--apple-blue)';
        else if (overall >= 60) strokeEl.style.stroke = 'var(--warning)';
        else strokeEl.style.stroke = 'var(--danger)';
    }

    if (ratingEl) {
        ratingEl.innerText = sc.readiness_rating || 'Competent Submission';
        if (overall >= 85) {
            ratingEl.style.borderColor = 'rgba(74, 222, 128, 0.4)';
            ratingEl.style.color = 'var(--success)';
            if (subtextEl) subtextEl.innerText = 'Bench inclined to issue notice / Rule Nisi';
        } else if (overall >= 74) {
            ratingEl.style.borderColor = 'rgba(201, 169, 97, 0.4)';
            ratingEl.style.color = 'var(--apple-azure)';
            if (subtextEl) subtextEl.innerText = 'Survives preliminary cross-examination';
        } else {
            ratingEl.style.borderColor = 'rgba(212, 175, 106, 0.4)';
            ratingEl.style.color = 'var(--warning)';
            if (subtextEl) subtextEl.innerText = 'Requires binding authority reinforcement';
        }
    }

    // 4 Dimension Bars
    const setMetric = (valId, barId, val) => {
        const vEl = document.getElementById(valId);
        const bEl = document.getElementById(barId);
        const v = Math.round(val || 75);
        if (vEl) vEl.innerText = `${v}%`;
        if (bEl) bEl.style.width = `${v}%`;
    };
    setMetric('val-grounding', 'bar-grounding', sc.constitutional_grounding);
    setMetric('val-statutory', 'bar-statutory', sc.statutory_precision);
    setMetric('val-precedent', 'bar-precedent', sc.precedent_authority);
    setMetric('val-persuasion', 'bar-persuasion', sc.judicial_persuasion || 74);

    // Strengths and Vulnerabilities Lists
    const strList = document.getElementById('moot-strengths-list');
    if (strList && data.strengths) {
        strList.innerHTML = data.strengths.map(s => `<li>${s}</li>`).join('');
    }
    const vulList = document.getElementById('moot-vulnerabilities-list');
    if (vulList && data.vulnerabilities) {
        vulList.innerHTML = data.vulnerabilities.map(v => `<li>${v}</li>`).join('');
    }

    // Rebuttal Tip and Counter-Precedent
    const rebEl = document.getElementById('moot-rebuttal-tip');
    if (rebEl) rebEl.innerText = data.rebuttal_tip || 'Anchor relief on express statutory wording.';

    const cpWrap = document.getElementById('moot-counter-precedent-wrap');
    if (cpWrap) cpWrap.style.display = 'block';

    const cpEl = document.getElementById('moot-counter-precedent');
    if (cpEl) cpEl.innerText = data.counter_precedent || 'Binding Supreme Court Precedent';

    const cprEl = document.getElementById('moot-counter-precedent-ratio');
    if (cprEl) cprEl.innerText = data.counter_precedent_ratio || 'Statutory bars are strictly enforced unless constitutional unreasonableness is demonstrated.';
}

// 7. Speech Synthesis ("Hear the Bench") with Server Stream & Web Speech Fallback
window.speakMootMessage = async function(btn) {
    const bubble = btn.closest('.moot-bubble');
    if (!bubble) return;
    const text = bubble.querySelector('p')?.innerText || '';
    if (!text) return;

    // If currently playing audio, stop it
    if (activeTTSAudio) {
        activeTTSAudio.pause();
        activeTTSAudio = null;
        if (window.speechSynthesis) window.speechSynthesis.cancel();
        btn.classList.remove('is-playing');
        btn.innerHTML = '<i class="ri-volume-up-line"></i> <span>Hear the Bench</span>';
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<i class="ri-loader-4-line spin"></i> <span>Synthesizing...</span>';

    // 1. Try server audio streaming via /api/speech/speak
    try {
        const res = await fetch(apiUrl('/api/speech/speak'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, language: state.lang || 'en' })
        });

        if (res.ok) {
            const blob = await res.blob();
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            activeTTSAudio = audio;

            btn.disabled = false;
            btn.classList.add('is-playing');
            btn.innerHTML = '<i class="ri-volume-vibrate-line"></i> <span>Playing Bench...</span>';

            audio.onended = () => {
                btn.classList.remove('is-playing');
                btn.innerHTML = '<i class="ri-volume-up-line"></i> <span>Hear the Bench</span>';
                URL.revokeObjectURL(audioUrl);
                activeTTSAudio = null;
            };
            audio.onerror = () => {
                playWebSpeechFallback(text, btn);
            };
            await audio.play();
            return;
        }
    } catch (e) {
        console.warn('Server TTS stream unavailable, falling back to Web Speech API:', e);
    }

    // 2. Client-side Web Speech API Fallback
    playWebSpeechFallback(text, btn);
};

function playWebSpeechFallback(text, btn) {
    if (!('speechSynthesis' in window)) {
        btn.disabled = false;
        btn.innerHTML = '<i class="ri-volume-up-line"></i> <span>Hear the Bench</span>';
        showToast('Speech audio playback is not supported in this browser.', 'warning');
        return;
    }

    window.speechSynthesis.cancel();
    const cleanText = text.replace(/[*_#`]/g, '');
    const utter = new SpeechSynthesisUtterance(cleanText);
    utter.rate = 0.95;
    utter.pitch = 0.92;
    utter.lang = state.lang || 'en-IN';

    btn.disabled = false;
    btn.classList.add('is-playing');
    btn.innerHTML = '<i class="ri-volume-vibrate-line"></i> <span>Playing Bench...</span>';

    utter.onend = () => {
        btn.classList.remove('is-playing');
        btn.innerHTML = '<i class="ri-volume-up-line"></i> <span>Hear the Bench</span>';
        activeTTSAudio = null;
    };
    utter.onerror = () => {
        btn.classList.remove('is-playing');
        btn.innerHTML = '<i class="ri-volume-up-line"></i> <span>Hear the Bench</span>';
        activeTTSAudio = null;
    };

    window.speechSynthesis.speak(utter);
}

// 8. Session Reset & Minutes Export
window.resetMootSession = function() {
    if (confirm("Reset current Moot Court hearing? This will clear oral argument history and return to preliminary stage.")) {
        onBenchSelectChange();
        resetMootTimer();
        showToast("Moot Court hearing reset to Round 1.", "info");
    }
};

window.exportMootSession = async function() {
    const benchSelect = document.getElementById('moot-bench-select');
    const tempSelect = document.getElementById('moot-temperament-select');
    const bType = benchSelect ? benchSelect.value : 'constitutional';
    const temperament = tempSelect ? tempSelect.value : 'inquisitive';
    const sideSelect = document.getElementById('moot-side-select');
    const counselSide = sideSelect ? sideSelect.value : 'petitioner';
    const probSelect = document.getElementById('moot-problem-select');
    const probKey = probSelect ? probSelect.value : 'open';
    const prob = (probKey && probKey !== 'open') ? MOOT_PROBLEMS_CATALOG[probKey] : null;
    const caseTitle = prob ? prob.title : `Supreme Court Special Appellate Hearing - ${bType.toUpperCase()}`;

    const rounds = state.mootHistory && state.mootHistory.length > 0
        ? state.mootHistory
        : [{ counsel: "Preliminary oral proposition.", bench: "Preliminary cross-examination." }];

    const scorecard = {
        overall_score: document.getElementById('moot-overall-score')?.innerText || '80',
        readiness_rating: document.getElementById('moot-readiness-rating')?.innerText || 'Competent Submission',
        constitutional_grounding: document.getElementById('val-grounding')?.innerText?.replace('%', '') || '85',
        statutory_precision: document.getElementById('val-statutory')?.innerText?.replace('%', '') || '75',
        precedent_authority: document.getElementById('val-precedent')?.innerText?.replace('%', '') || '80',
        judicial_persuasion: document.getElementById('val-persuasion')?.innerText?.replace('%', '') || '78',
        rebuttal_tip: document.getElementById('moot-rebuttal-tip')?.innerText || ''
    };

    try {
        const res = await api('/api/moot/export', {
            method: 'POST',
            body: JSON.stringify({
                bench_type: bType,
                temperament: temperament,
                rounds: rounds,
                scorecard: scorecard,
                case_title: caseTitle,
                counsel_side: counselSide
            })
        });

        const record = res.record || 'Moot Court Session Record';
        const blob = new Blob([record], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Supreme_Court_Order_Sheet_${bType}_Round${state.mootRound || 1}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast("Official Supreme Court Order Sheet exported successfully.", "success");
    } catch (err) {
        console.error("Export minutes error:", err);
        showToast("Failed to export court minutes.", "error");
    }
};

// 9. Miscellaneous Helper Actions
window.copyBenchPrompt = function(btn) {
    const bubble = btn.closest('.moot-bubble');
    if (!bubble) return;
    const text = bubble.querySelector('p')?.innerText || '';
    if (text) {
        navigator.clipboard.writeText(text);
        showToast("Judicial interjection copied to clipboard.", "success");
    }
};

window.toggleCompendiumAccordion = function() {
    const body = document.getElementById('moot-compendium-body');
    const icon = document.getElementById('compendium-arrow-icon');
    if (!body) return;
    if (body.style.display === 'none') {
        body.style.display = 'flex';
        if (icon) icon.className = 'ri-arrow-down-s-line';
    } else {
        body.style.display = 'none';
        if (icon) icon.className = 'ri-arrow-right-s-line';
    }
};

/* ── 5. Case Dossier & Court Brief Generator ────────────────────────────── */

window.generateDossierBrief = async function() {
    const court = document.getElementById('dossier-court')?.value || 'IN THE SUPREME COURT OF INDIA';
    const title = document.getElementById('dossier-title')?.value || 'Cause Title';
    const query = document.getElementById('dossier-query')?.value || 'Legal Question';
    const facts = document.getElementById('dossier-facts')?.value || 'Factual matrix';

    if (!query.trim() && !facts.trim()) {
        alert('Please enter the core legal proposition and factual matrix.');
        return;
    }

    try {
        const res = await api('/api/dossier/generate', {
            method: 'POST',
            body: JSON.stringify({
                case_title: title,
                query: query,
                answer: facts,
                court: court,
                citations: ['(2024) INSC 595', '(2022) 10 SCC 51']
            })
        });

        const d = res.dossier || {};
        const chEl = document.getElementById('pv-court-heading');
        if (chEl) chEl.innerText = d.court_heading || court;
        const ctEl = document.getElementById('pv-cause-title');
        if (ctEl) ctEl.innerText = d.case_title || title;

        // Populate Questions
        const qList = document.getElementById('pv-questions-list');
        if (qList && (d.questions_framed || d.issues_framed)) {
            const issues = d.questions_framed || d.issues_framed;
            qList.innerHTML = issues.map(q => `<li>${escapeHtml(q)}</li>`).join('');
        }

        // Populate Table of Authorities
        const authTable = document.getElementById('pv-table-authorities')?.querySelector('tbody');
        if (authTable && d.table_of_authorities) {
            authTable.innerHTML = d.table_of_authorities.map(a => `
                <tr>
                    <td><strong>${escapeHtml(a.case_name || a.title || 'Precedent')}</strong></td>
                    <td>${escapeHtml(a.citation || '')}</td>
                    <td>${escapeHtml(a.ratio || a.holding || '')}</td>
                </tr>
            `).join('');
        }

        // Populate Statutory Matrix
        const statMatrix = document.getElementById('pv-statutory-matrix');
        if (statMatrix && d.statutory_matrix) {
            statMatrix.innerHTML = d.statutory_matrix.map(s => `
                <p><strong>${escapeHtml(s.act)}:</strong> ${escapeHtml(s.provisions)}</p>
            `).join('');
        }

        // Populate Submissions
        const subBody = document.getElementById('pv-submissions-body');
        if (subBody) {
            subBody.innerText = d.synopsis || d.executive_summary || facts;
        }
    } catch (e) {
        console.error('Dossier generation error:', e);
        alert(`Error generating dossier: ${e.message}`);
    }
};

window.exportDossierToPdf = async function() {
    const court = document.getElementById('dossier-court')?.value || 'IN THE SUPREME COURT OF INDIA';
    const title = document.getElementById('dossier-title')?.value || 'Cause Title';
    const query = document.getElementById('dossier-query')?.value || 'Legal Question';
    const facts = document.getElementById('dossier-facts')?.value || 'Factual matrix';
    const btn = document.getElementById('btn-export-pdf');

    try {
        if (btn) btn.disabled = true;
        btn.innerHTML = '<i class="ri-loader-4-line spin"></i> Generating Court PDF...';

        const res = await fetch(apiUrl('/api/dossier/export-pdf'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                case_title: title,
                query: query,
                answer: facts,
                court: court,
                citations: ['(2024) INSC 595', '(2022) 10 SCC 51']
            })
        });

        if (!res.ok) throw new Error(`Export failed (HTTP ${res.status})`);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `LegalMind_Court_Brief_${Date.now()}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) {
        console.error('PDF export error:', e);
        alert(`Failed to export court PDF: ${e.message}`);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="ri-file-pdf-line"></i> <span>Download Court-Ready PDF</span>';
        }
    }
};


/* ── 6. New AI Suite Handlers ────────────────────────────── */

window.runTemporalConvert = async function() {
    const d = document.getElementById('temp-date')?.value || '';
    const a = document.getElementById('temp-act')?.value || 'ipc';
    const s = document.getElementById('temp-sections')?.value || '';
    const resDiv = document.getElementById('temporal-results');
    if (!resDiv) return;

    if (!d.trim()) {
        resDiv.innerHTML = '<p style="color:var(--danger)"><i class="ri-error-warning-line"></i> Please select the date of the alleged offense.</p>';
        return;
    }
    const secList = s.split(',').map(x => x.trim()).filter(x => x);
    if (!secList.length) {
        resDiv.innerHTML = '<p style="color:var(--danger)"><i class="ri-error-warning-line"></i> Please enter at least one section number (e.g. 420, 506).</p>';
        return;
    }

    try {
        resDiv.innerHTML = '<p><i class="ri-loader-4-line spin"></i> Converting provisions across temporal cutoff...</p>';
        const res = await api('/api/temporal/convert', {
            method: 'POST',
            body: JSON.stringify({
                offense_date: d,
                act: a,
                sections: secList
            })
        });
        
        let html = `<h3>${escapeHtml(res.regime)}</h3><p><strong>Rationale:</strong> ${escapeHtml(res.regime_rationale)}</p>`;
        if (res.temporal_warning) html += `<div style="color:var(--danger); margin:10px 0;">${escapeHtml(res.temporal_warning)}</div>`;
        html += `<table class="paper-table"><thead><tr><th>Original</th><th>New Section</th><th>Status</th><th>Notes</th></tr></thead><tbody>`;
        (res.section_mappings || []).forEach(m => {
            html += `<tr><td>${escapeHtml(m.original_section)}</td><td>${escapeHtml(m.new_section)}</td><td>${escapeHtml(m.status)}</td><td>${escapeHtml(m.change_summary)}</td></tr>`;
        });
        html += `</tbody></table>`;
        resDiv.innerHTML = html;
    } catch (e) {
        resDiv.innerHTML = `<p style="color:var(--danger)">Error: ${escapeHtml(e.message)}</p>`;
    }
};

window.runBailMatrix = async function() {
    const cat = document.getElementById('bail-category')?.value || 'Economic Offence';
    const sp = !!document.getElementById('bail-special-act')?.checked;
    const custRaw = parseInt(document.getElementById('bail-custody')?.value, 10);
    const cust = Number.isFinite(custRaw) && custRaw > 0 ? custRaw : 0;
    const chg = !!document.getElementById('bail-chargesheet')?.checked;
    const resDiv = document.getElementById('bail-results');
    if (!resDiv) return;
    const list = items => (items || []).map(x => `<li>${escapeHtml(x)}</li>`).join('') || '<li>None</li>';

    try {
        resDiv.innerHTML = '<p><i class="ri-loader-4-line spin"></i> Assessing statutory bail feasibility...</p>';
        const res = await api('/api/bail/assess', {
            method: 'POST',
            body: JSON.stringify({
                offense_category: cat,
                is_special_act: sp,
                custody_days: cust,
                chargesheet_filed: chg
            })
        });
        const d = res.assessment;
        let html = `<h3>Verdict: ${escapeHtml(d.verdict)} (Score: ${escapeHtml(d.bail_score)}/100)</h3>`;
        html += `<h4>Positive Factors</h4><ul>${list(d.positive_factors)}</ul>`;
        html += `<h4>Risk Factors</h4><ul>${list(d.risk_factors)}</ul>`;
        html += `<h4>Precedents</h4><ul>${list(d.relevant_precedents)}</ul>`;
        resDiv.innerHTML = html;
    } catch (e) {
        resDiv.innerHTML = `<p style="color:var(--danger)">Error: ${escapeHtml(e.message)}</p>`;
    }
};

window.runFirAudit = async function() {
    const text = document.getElementById('fir-text')?.value || '';
    const arr = !!document.getElementById('fir-arrest')?.checked;
    const pmla = !!document.getElementById('fir-pmla')?.checked;
    const resDiv = document.getElementById('fir-results');
    if (!resDiv) return;

    if (!text.trim()) {
        resDiv.innerHTML = '<p style="color:var(--danger)"><i class="ri-error-warning-line"></i> Error: FIR text cannot be empty. Please enter or paste the FIR narrative.</p>';
        return;
    }
    
    try {
        resDiv.innerHTML = '<p><i class="ri-loader-4-line spin"></i> Performing procedural screening...</p>';
        const res = await api('/api/fir/audit', {
            method: 'POST',
            body: JSON.stringify({ fir_text: text, arrest_made: arr, is_pmla: pmla })
        });
        const d = res ? (res.audit_report || res) : null;
        const esc = (t) => { const n = document.createElement('div'); n.textContent = t == null ? '' : String(t); return n.innerHTML; };

        if (!d || d.status === 'error') {
            resDiv.innerHTML = `<p style="color:var(--danger)">${esc(d ? d.message || d.error : 'Error running procedural screening.')}</p>`;
            return;
        }

        if (d.status === 'insufficient_input') {
            resDiv.innerHTML = `
                <div style="background: rgba(245, 158, 11, 0.12); border-left: 4px solid var(--warning, #f59e0b); padding: 12px 16px; border-radius: 6px; margin-bottom: 14px;">
                    <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: var(--warning, #f59e0b);">Screening Status</div>
                    <div style="font-size: 16px; font-weight: 700; margin-top: 2px;">${esc(d.overall_status || 'Insufficient Information')}</div>
                    <p style="font-size: 13px; margin-top: 6px; opacity: 0.9;">${esc(d.message)}</p>
                </div>
                ${d.documents_required ? `<h4>Recommended Material to Supply:</h4><ul>${d.documents_required.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}
            `;
            return;
        }

        const rf = d.read_from_fir || {};
        let html = `<h3>${esc(d.screening_header || 'Preliminary Procedural Screening')}</h3>`;

        // Overall status badge
        const isAmber = (d.overall_status || '').includes('Potential') || (d.overall_status || '').includes('Significant');
        const statusColor = isAmber ? 'var(--warning, #f59e0b)' : 'var(--accent, #38bdf8)';
        html += `
            <div style="background: rgba(245, 158, 11, 0.08); border-left: 4px solid ${statusColor}; padding: 12px 16px; border-radius: 6px; margin: 12px 0 16px 0;">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: ${statusColor};">Overall Status</div>
                <div style="font-size: 16px; font-weight: 700; margin-top: 2px; color: ${statusColor};">${esc(d.overall_status)}</div>
                <div style="font-size: 13px; margin-top: 6px; opacity: 0.9; line-height: 1.4;">${esc(d.status_explanation || '')}</div>
            </div>
        `;

        // Applicable provision
        html += `
            <div style="margin-bottom: 16px; font-size: 13px; background: rgba(255, 255, 255, 0.04); padding: 10px 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
                <strong style="color:var(--text-muted, #94a3b8); font-size: 11px; text-transform: uppercase;">Applicable Provision</strong><br>
                <div style="color: var(--accent); font-weight: 600; margin-top: 3px; white-space: pre-line;">${esc(d.applicable_provision)}</div>
            </div>
        `;

        // Read from this FIR
        html += `<h4>Read from this FIR</h4>`;
        html += `<ul style="list-style-type: none; padding-left: 0; margin-bottom: 18px; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px;">`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Alleged occurrence:</strong> ${esc(rf.alleged_occurrence)}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• FIR registered:</strong> ${esc(rf.fir_registered)}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Registration delay:</strong> ${esc(rf.registration_delay)}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Reason for delay:</strong> ${esc(rf.delay_reason)}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Sections invoked:</strong> ${esc(rf.sections_invoked)}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Alleged offences:</strong> ${esc(rf.alleged_offences || 'None specified')}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Accused:</strong> ${esc(rf.accused || rf.accused_count || 'Not stated')}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Arrest:</strong> ${esc(rf.arrest)}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Special statute:</strong> ${esc(rf.special_statute)}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Property/ownership dispute:</strong> ${esc(rf.property_dispute || 'Not mentioned')}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Contractual dispute:</strong> ${esc(rf.contractual_dispute)}</li>`;
        html += `<li style="background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 4px; font-size: 12px;"><strong>• Settlement/repayment:</strong> ${esc(rf.settlement_repayment)}</li>`;
        html += `</ul>`;

        // Procedural Findings
        html += `<h4>Procedural Findings</h4>`;
        if (!(d.procedural_findings || []).length) {
            html += `<p style="font-size: 13px; opacity: 0.8;">No apparent procedural defect identified from supplied material.</p>`;
        } else {
            html += `<div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 20px;">` + d.procedural_findings.map((x, idx) => {
                const st = (x.status || '').toUpperCase();
                let badgeColor = 'var(--accent, #38bdf8)';
                let badgeBg = 'rgba(56, 189, 248, 0.12)';
                if (st.includes('POTENTIAL') || st.includes('AMBER')) {
                    badgeColor = 'var(--warning, #f59e0b)';
                    badgeBg = 'rgba(245, 158, 11, 0.12)';
                } else if (st.includes('VERIFICATION')) {
                    badgeColor = '#fb923c';
                    badgeBg = 'rgba(251, 146, 60, 0.12)';
                } else if (st.includes('NOT_APPLICABLE') || st.includes('NOT APPLICABLE')) {
                    badgeColor = 'var(--text-muted, #94a3b8)';
                    badgeBg = 'rgba(148, 163, 184, 0.12)';
                } else if (st.includes('COMPLIANCE')) {
                    badgeColor = 'var(--success, #10b981)';
                    badgeBg = 'rgba(16, 185, 129, 0.12)';
                }
                const verStr = Array.isArray(x.verification_required) ? x.verification_required.join(', ') : (x.verification_required || '');
                return `
                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 6px; padding: 12px 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
                            <div style="font-weight: 700; font-size: 14px;">${idx + 1}. ${esc(x.title)}</div>
                            <span style="background: ${badgeBg}; color: ${badgeColor}; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; text-transform: uppercase;">${esc(x.status)}</span>
                        </div>
                        <div style="font-size: 12px; margin-top: 4px; background: rgba(0,0,0,0.15); padding: 6px 10px; border-radius: 4px;"><strong>Read from FIR:</strong> "${esc(x.read_from_fir)}"</div>
                        <div style="font-size: 12px; margin-top: 6px; opacity: 0.9;"><strong>Legal principle:</strong> ${esc(x.legal_principle)}</div>
                        ${x.authority ? `<div style="font-size: 12px; margin-top: 4px; color: var(--accent);"><strong>Authority:</strong> ${esc(x.authority)}</div>` : ''}
                        <div style="font-size: 12px; margin-top: 4px; opacity: 0.9;"><strong>Assessment:</strong> ${esc(x.assessment)}</div>
                        ${verStr ? `<div style="font-size: 12px; margin-top: 4px; opacity: 0.85;"><strong>Verification required:</strong> ${esc(verStr)}</div>` : ''}
                        <div style="font-size: 11px; margin-top: 6px; opacity: 0.7;"><strong>Confidence:</strong> ${esc(x.confidence)} ${x.confidence_explanation ? `(${esc(x.confidence_explanation)})` : ''}</div>
                    </div>
                `;
            }).join('') + `</div>`;
        }

        // Ingredient Analysis
        if ((d.ingredient_analysis || []).length) {
            html += `<h4>Offence Ingredient Analysis</h4>`;
            html += `<div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">` + d.ingredient_analysis.map(off => `
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 10px 14px;">
                    <div style="font-weight: 700; font-size: 13px; color: var(--accent); margin-bottom: 6px;">
                        ${esc(off.offence_name)} <span style="opacity: 0.75; font-weight: normal;">(${esc(off.invoked_section)} → ${esc(off.corresponding_bns)})</span>
                    </div>
                    <ul style="padding-left: 18px; margin: 6px 0; font-size: 12px;">
                        ${(off.ingredients || []).map(ing => `
                            <li style="margin-bottom: 4px;">
                                <strong>Ingredient ${ing.ingredient_number}:</strong> ${esc(ing.ingredient_text)}<br>
                                <span style="opacity: 0.85;">FIR Support: ${esc(ing.fir_support)}</span> — 
                                <span style="font-weight: 600; color: ${ing.status === 'Stated' ? 'var(--success, #10b981)' : 'var(--warning, #f59e0b)'};">[${esc(ing.status)}]</span>
                            </li>
                        `).join('')}
                    </ul>
                    <div style="font-size: 11px; opacity: 0.8; font-style: italic; margin-top: 4px;">${esc(off.ingredient_assessment)}</div>
                </div>
            `).join('') + `</div>`;
        }

        // Potential Inherent-Powers / Quashing-Relevant Issues
        if ((d.quashing_issues || []).length) {
            html += `<h4>Potential Inherent-Powers / Quashing-Relevant Issues</h4><ul style="font-size: 13px; line-height: 1.5; margin-bottom: 18px;">` +
                d.quashing_issues.map(qi => `<li>${esc(qi)}</li>`).join('') + `</ul>`;
        }

        // Material Information Not Established
        if ((d.material_not_established || []).length) {
            html += `<h4>Material Information Not Established</h4><ul style="font-size: 12px; line-height: 1.5; margin-bottom: 18px; opacity: 0.85;">` +
                d.material_not_established.map(m => `<li>${esc(m)}</li>`).join('') + `</ul>`;
        }

        // Documents recommended for verification
        if ((d.documents_recommended || []).length) {
            html += `<h4>Documents Recommended for Verification</h4><ul style="font-size: 12px; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 4px; padding-left: 18px; margin-bottom: 18px;">` +
                d.documents_recommended.map(doc => `<li>${esc(doc)}</li>`).join('') + `</ul>`;
        }

        // Deduplicated retrieved sources
        if ((d.retrieved_sources || []).length) {
            html += `<h4>Authoritative Sources Retrieved</h4>`;
            html += `<div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 18px;">` + d.retrieved_sources.map(s => `
                <div style="background: rgba(255, 255, 255, 0.02); border-left: 3px solid var(--accent); padding: 8px 12px; border-radius: 4px; font-size: 12px;">
                    <div style="font-weight: 700;">${esc(s.title || 'Legal Authority')} ${s.citation ? `<span style="font-weight: normal; opacity: 0.85;">— <em>${esc(s.citation)}</em></span>` : ''}</div>
                    <div style="font-size: 11px; opacity: 0.8; margin-top: 2px;"><strong>Authority:</strong> ${esc(s.court_or_authority)} | <strong>Provision:</strong> ${esc(s.relevant_section)}</div>
                    <div style="font-size: 12px; margin-top: 4px; opacity: 0.9;"><strong>Proposition:</strong> ${esc(s.legal_proposition)}</div>
                    <div style="font-size: 11px; color: var(--accent); opacity: 0.9; margin-top: 2px;"><strong>Relevance:</strong> ${esc(s.relevance_reason)}</div>
                </div>
            `).join('') + `</div>`;
        }

        // Cautionary disclaimer
        html += `<div style="font-size: 11px; opacity: 0.7; margin-top: 18px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px; line-height: 1.4;"><i class="ri-shield-check-line"></i> ${esc(d.disclaimer || '')}</div>`;

        resDiv.innerHTML = html;
    } catch (e) {
        const esc = (t) => { const n = document.createElement('div'); n.textContent = t == null ? '' : String(t); return n.innerHTML; };
        const msg = (e.message === 'Failed to fetch' || (e.message && e.message.includes('fetch')))
            ? 'Connection error or server temporarily unavailable. Please retry in a moment.'
            : e.message;
        resDiv.innerHTML = `<p style="color:var(--danger)"><i class="ri-error-warning-line"></i> Error: ${esc(msg)}</p>`;
    }
};

window.runCitationCheck = async function() {
    const text = document.getElementById('cit-text')?.value || '';
    const resDiv = document.getElementById('citcheck-results');
    if (!resDiv) return;

    if (!text.trim()) {
        resDiv.innerHTML = '<p style="color:var(--danger)"><i class="ri-error-warning-line"></i> Please enter or paste text containing citations or case names.</p>';
        return;
    }
    
    try {
        resDiv.innerHTML = '<p><i class="ri-loader-4-line spin"></i> Validating citation authority against Supreme Court database...</p>';
        const res = await api('/api/citations/validate', {
            method: 'POST',
            body: JSON.stringify({ text: text })
        });
        let html = `<div style="margin-bottom: 20px;">
            <h4>Input:</h4>
            <div style="padding: 10px; background: rgba(0,0,0,0.05); border-left: 3px solid var(--accent);">${escapeHtml(res.input_text || '')}</div>
            <h4 style="margin-top:10px;">Extracted Citations:</h4>
            <ul>${(res.extracted_citations || []).map(c => `<li>${escapeHtml(c)}</li>`).join('')}</ul>
        </div>`;
        
        (res.validations || []).forEach((v, idx) => {
            let color = 'var(--text)';
            if (v.status === 'VERIFIED') color = 'var(--accent)';
            else if (v.status === 'PARTIALLY VERIFIED') color = 'var(--warning)';
            else if (v.status === 'UNVERIFIED' || v.status === 'NOT FOUND' || v.status === 'INVALID') color = 'var(--danger)';
            
            html += `<div class="glass-panel" style="margin-bottom: 20px;">
                <h3 style="margin-bottom: 10px;">${idx + 1}. ${escapeHtml(v.exact_citation)}</h3>
                <table class="paper-table" style="width: 100%; text-align: left;">
                    <tbody>
                        <tr><th style="width: 25%;">Status</th><td style="color:${color}; font-weight:bold;">${escapeHtml(v.status)}</td></tr>
                        <tr><th>Matched Authority</th><td>${escapeHtml(v.case_or_statute)}</td></tr>
                        <tr><th>Court / Source</th><td>${escapeHtml(v.court)}</td></tr>
                        <tr><th>Citation Details</th><td>Year: ${escapeHtml(v.year)}, Reporter: ${escapeHtml(v.reporter_reference)}</td></tr>
                        <tr><th>Explanation</th><td>${escapeHtml(v.explanation)}</td></tr>
                        <tr><th>Source ID</th><td>${escapeHtml(v.source_id)}</td></tr>
                    </tbody>
                </table>
                <div style="margin-top: 10px; padding: 10px; background: rgba(240,230,216,0.1); border-left: 3px solid ${color};">
                    <strong>Evidence:</strong>
                    <p style="font-size: 0.9em; margin-top: 5px;">${escapeHtml(v.evidence)}</p>
                </div>
            </div>`;
        });
        resDiv.innerHTML = html;
    } catch (e) {
        resDiv.innerHTML = `<p style="color:var(--danger)">Error: ${escapeHtml(e.message)}</p>`;
    }
};

window.runLimitation = async function() {
    const t = document.getElementById('lim-type')?.value || 'money_suit';
    const d = document.getElementById('lim-date')?.value || '';
    const excRaw = parseInt(document.getElementById('lim-exclude')?.value, 10);
    const exc = Number.isFinite(excRaw) && excRaw >= 0 ? excRaw : 0;
    const resDiv = document.getElementById('limitation-results');
    if (!resDiv) return;

    if (!d.trim()) {
        resDiv.innerHTML = '<p style="color:var(--danger)"><i class="ri-error-warning-line"></i> Please select the date when the cause of action arose or impugned order was passed.</p>';
        return;
    }
    
    try {
        resDiv.innerHTML = '<p><i class="ri-loader-4-line spin"></i> Calculating statutory limitation deadline...</p>';
        const res = await api('/api/limitation/calculate', {
            method: 'POST',
            body: JSON.stringify({ suit_type: t, cause_of_action_date: d, exclude_days: exc })
        });
        const data = res.limitation_report;
        let color = data.status === 'EXPIRED' ? 'var(--danger)' : 'var(--accent)';
        let html = `<h3>Status: <span style="color:${color}">${escapeHtml(data.status)}</span></h3>`;
        html += `<p><strong>Statutory Deadline:</strong> ${escapeHtml(data.expiry_date)} (${escapeHtml(data.period_statute)})</p>`;
        if (data.status === 'EXPIRED') html += `<p><strong>Delay:</strong> ${escapeHtml(data.days_delayed)} days</p>`;
        else html += `<p><strong>Days Remaining:</strong> ${escapeHtml(data.days_remaining)} days</p>`;
        if (data.condonation_advice) html += `<div style="padding:10px; background:rgba(240,230,216,0.05); margin-top:10px; border-left: 3px solid var(--warning);">${escapeHtml(data.condonation_advice)}</div>`;
        resDiv.innerHTML = html;
    } catch (e) {
        resDiv.innerHTML = `<p style="color:var(--danger)">Error: ${escapeHtml(e.message)}</p>`;
    }
};
window.runLimitationCalc = window.runLimitation;


window.runPleading = async function() {
    const t = document.getElementById('pleading-type')?.value || 'bail';
    const txt = document.getElementById('pleading-text')?.value || '';
    const resDiv = document.getElementById('pleading-results');
    if (!resDiv) return;

    if (!txt.trim()) {
        resDiv.innerHTML = '<p style="color:var(--danger)"><i class="ri-error-warning-line"></i> Please dictate or enter case facts and lawyer notes to format.</p>';
        return;
    }
    
    try {
        resDiv.innerHTML = '<p><i class="ri-loader-4-line spin"></i> Formatting Pleading into formal Court draft...</p>';
        const res = await api('/api/pleading/format', {
            method: 'POST',
            body: JSON.stringify({ pleading_type: t, raw_text: txt })
        });
        const draftText = res.pleading_draft ? res.pleading_draft.formatted_draft : '';
        const html = `<div class="paper-text-body" style="white-space: pre-wrap; font-family: 'Times New Roman', serif; background: var(--bg-surface); border: 1px solid var(--border-glass); padding: 20px; border-radius: 8px;">${escapeHtml(draftText)}</div>`;
        resDiv.innerHTML = html;
    } catch (e) {
        resDiv.innerHTML = `<p style="color:var(--danger)">Error: ${escapeHtml(e.message)}</p>`;
    }
};

let pleadingRecognition = null;
window.togglePleadingDictation = function() {
    const btn = document.getElementById('pleading-mic-btn');
    const label = document.getElementById('pleading-mic-text');
    const textarea = document.getElementById('pleading-text');
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRec) {
        alert('Voice dictation requires browser SpeechRecognition (e.g. Chrome or Edge).');
        return;
    }

    if (pleadingRecognition) {
        pleadingRecognition.stop();
        pleadingRecognition = null;
        if (btn) btn.classList.remove('recording');
        if (label) label.innerText = 'Start Dictation';
        return;
    }

    try {
        pleadingRecognition = new SpeechRec();
        pleadingRecognition.continuous = true;
        pleadingRecognition.interimResults = true;
        pleadingRecognition.lang = 'en-IN';

        pleadingRecognition.onstart = () => {
            if (btn) btn.classList.add('recording');
            if (label) label.innerText = 'Listening... (Click to stop)';
        };

        pleadingRecognition.onresult = (e) => {
            let transcript = '';
            for (let i = 0; i < e.results.length; i++) {
                transcript += e.results[i][0].transcript + ' ';
            }
            if (textarea) textarea.value = transcript.trim();
        };

        pleadingRecognition.onerror = (err) => {
            console.error('Dictation error:', err);
            if (btn) btn.classList.remove('recording');
            if (label) label.innerText = 'Start Dictation';
            pleadingRecognition = null;
        };

        pleadingRecognition.onend = () => {
            if (btn) btn.classList.remove('recording');
            if (label) label.innerText = 'Start Dictation';
            pleadingRecognition = null;
        };

        pleadingRecognition.start();
    } catch (err) {
        console.error('Voice dictation failed to start:', err);
        alert(`Microphone dictation error: ${err.message}`);
    }
};

/* ── 7. Theme Switching (Light / Dark Court Modes) ───────────── */
window.toggleTheme = function() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try {
        localStorage.setItem('legalmind_theme', next);
    } catch (e) {}
    updateThemeIcon(next);
};

function updateThemeIcon(theme) {
    const icon = document.getElementById('theme-toggle-icon');
    const btn = document.getElementById('theme-toggle-btn');
    if (icon) {
        if (theme === 'dark') {
            icon.className = 'ri-sun-line';
            if (btn) btn.title = 'Switch to Light Court Mode';
        } else {
            icon.className = 'ri-moon-line';
            if (btn) btn.title = 'Switch to Dark Court Mode';
        }
    }
}

// Initialize theme icon on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        const theme = document.documentElement.getAttribute('data-theme') || 'light';
        updateThemeIcon(theme);
    });
} else {
    const theme = document.documentElement.getAttribute('data-theme') || 'light';
    updateThemeIcon(theme);
}



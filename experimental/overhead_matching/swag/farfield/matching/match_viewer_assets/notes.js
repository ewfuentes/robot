// Free-form human notes live outside immutable artifacts in one data-root
// sidecar.  A plain static server can show an existing note by reading that
// file; the dedicated viewer server adds the same-origin PUT endpoint.
(function(){
  const C = MATCH_NOTES_CONTEXT;
  const panel = document.getElementById('match-note-panel');
  const heading = document.getElementById('match-note-heading');
  const textarea = document.getElementById('match-note-text');
  const save = document.getElementById('match-note-save');
  const clear = document.getElementById('match-note-clear');
  const status = document.getElementById('match-note-status');
  const updated = document.getElementById('match-note-updated');
  const buttons = [...document.querySelectorAll('[data-note-select]')];
  let notes = {}, drafts = {}, dirty = new Set();
  let selected = null, writable = false;

  function record(key){ return notes[key] || null; }
  function visibleText(key){
    return dirty.has(key) ? drafts[key] : ((record(key) || {}).text || '');
  }
  function refreshButtons(){
    buttons.forEach(button => {
      const key = button.dataset.noteSelect;
      const has = Boolean(visibleText(key).trim());
      button.classList.toggle('has-note', has);
      button.classList.toggle('unsaved-note', dirty.has(key));
      button.textContent = dirty.has(key) ? 'note (unsaved)'
        : has ? 'note' : 'add note';
      button.title = has ? visibleText(key) : 'Add a free-form human note';
    });
  }
  function refreshPanel(){
    const active = selected !== null;
    heading.textContent = active ? selected : 'Select a tracklet';
    textarea.disabled = !active || !writable;
    save.disabled = !active || !writable || !dirty.has(selected);
    clear.disabled = !active || !writable || !visibleText(selected).trim();
    if(!active){
      textarea.value = '';
      updated.textContent = '';
      return;
    }
    const text = visibleText(selected);
    // Avoid assigning the same value on every input event: doing so moves the
    // caret to the end and makes editing the middle of a note miserable.
    if(textarea.value !== text) textarea.value = text;
    const note = record(selected);
    updated.textContent = dirty.has(selected) ? 'unsaved changes'
      : note ? 'saved ' + note.updated_at : 'no note yet';
  }
  function selectNote(key){
    if(!key) return;
    selected = key;
    panel.classList.add('active');
    refreshPanel();
  }
  window.selectMatchNote = selectNote;

  async function responseJson(response){
    const body = await response.json().catch(() => ({}));
    if(!response.ok) throw new Error(body.error || 'HTTP ' + response.status);
    return body;
  }
  async function loadNotes(){
    const digest = encodeURIComponent(C.matching.content_digest);
    try {
      const body = await responseJson(await fetch(
        '/api/match-notes?matching_digest=' + digest, {cache:'no-store'}));
      notes = body.tracks || {};
      writable = true;
      status.textContent = 'central notes connected';
      status.className = 'note-status connected';
    } catch(apiError) {
      // `python -m http.server` has no API, but it can still expose the
      // centralized document read-only.  The dedicated server deliberately
      // hides this raw path and answers through the API above instead.
      try {
        const response = await fetch(
          '/_annotations/match_notes.json', {cache:'no-store'});
        if(!response.ok) throw new Error('HTTP ' + response.status);
        const document = await response.json();
        const run = (document.runs || {})[C.matching.content_digest];
        notes = run ? run.tracks || {} : {};
        status.textContent = 'notes are read-only under the static server';
        status.className = 'note-status readonly';
      } catch(staticError) {
        notes = {};
        status.textContent = 'start the farfield viewer server to use notes';
        status.className = 'note-status readonly';
      }
    }
    refreshButtons();
    refreshPanel();
  }
  async function saveSelected(text){
    if(!selected || !writable) return;
    const key = selected;
    save.disabled = true; clear.disabled = true;
    status.textContent = 'saving…';
    status.className = 'note-status';
    try {
      const body = await responseJson(await fetch('/api/match-notes', {
        method:'PUT',
        headers:{
          'Content-Type':'application/json',
          'X-Farfield-Viewer':'match-notes-v1'
        },
        body:JSON.stringify({matching:C.matching, tracklet_id:key, text:text})
      }));
      if(body.note) notes[key] = body.note; else delete notes[key];
      dirty.delete(key); delete drafts[key];
      status.textContent = body.note ? 'saved' : 'note cleared';
      status.className = 'note-status connected';
      refreshButtons(); refreshPanel();
    } catch(error) {
      status.textContent = 'save failed: ' + error.message;
      status.className = 'note-status failed';
      refreshPanel();
    }
  }

  buttons.forEach(button => button.addEventListener('click', event => {
    event.preventDefault(); event.stopPropagation();
    selectNote(button.dataset.noteSelect);
    panel.scrollIntoView({block:'nearest', behavior:'smooth'});
  }));
  document.querySelectorAll('.card h2').forEach(title => {
    title.addEventListener('click', () =>
      selectNote(title.parentElement.dataset.key));
  });
  textarea.addEventListener('input', () => {
    if(!selected) return;
    drafts[selected] = textarea.value;
    dirty.add(selected);
    refreshButtons(); refreshPanel();
  });
  textarea.addEventListener('keydown', event => {
    if((event.ctrlKey || event.metaKey) && event.key === 'Enter'){
      event.preventDefault(); saveSelected(textarea.value);
    }
  });
  save.addEventListener('click', () => saveSelected(textarea.value));
  clear.addEventListener('click', () => saveSelected(''));
  window.addEventListener('beforeunload', event => {
    if(!dirty.size) return;
    event.preventDefault(); event.returnValue = '';
  });
  if(location.hash){
    const key = decodeURIComponent(location.hash.slice(1));
    if(buttons.some(button => button.dataset.noteSelect === key))
      selectNote(key);
  }
  refreshButtons(); refreshPanel(); loadNotes();
})();

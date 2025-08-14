// static/app.js
const chatEl = document.getElementById("chat");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const clearBtn = document.getElementById("clear");
const exportBtn = document.getElementById("export");
const temperatureEl = document.getElementById("temperature");
const topPEl = document.getElementById("top_p");
const maxNewTokensEl = document.getElementById("max_new_tokens");

let history = [];

function bubble(role, text, pending=false) {
  const div = document.createElement("div");
  div.className = `msg ${role}${pending ? " pending" : ""}`;
  div.innerHTML = `
    <div class="role">${role}</div>
    <div class="meta"></div>
    <div class="text"></div>
  `;
  div.querySelector(".text").textContent = text || "";
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function setMeta(el, emotion, act) {
  const meta = el.querySelector(".meta");
  if (!meta) return;
  const safeEmo = (emotion || "").toString();
  const safeAct = (act || "").toString();
  meta.innerHTML = `
    <span class="tag">Emotion: ${safeEmo}</span>
    <span class="tag">Act: ${safeAct}</span>
  `;
}

async function sendMessage() {
  const message = promptEl.value.trim();
  if (!message) return;
  promptEl.value = "";

  history.push({ role: "user", content: message });
  bubble("user", message);

  const a = bubble("assistant", "", true);

  const payload = {
    message,
    history: history.slice(0, -1),
    temperature: parseFloat(temperatureEl.value || "0.7"),
    top_p: parseFloat(topPEl.value || "0.9"),
    max_new_tokens: parseInt(maxNewTokensEl.value || "256"),
  };

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok || !res.body) {
    a.classList.remove("pending");
    a.querySelector(".text").textContent = "[Error]";
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let partial = "";
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    partial += decoder.decode(value, { stream: true });
    const events = partial.split("\n\n");
    partial = events.pop() || "";
    for (const evt of events) {
      const lines = evt.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;

      try {
        const obj = JSON.parse(data);

        if (event === "meta") {
          setMeta(a, obj.emotion, obj.act);
        } else if (event === "delta") {
          buf += obj.text || "";
          a.querySelector(".text").textContent = buf;
          chatEl.scrollTop = chatEl.scrollHeight;
        } else if (event === "done") {
          a.classList.remove("pending");
          a.querySelector(".text").textContent = buf;
          history.push({ role: "assistant", content: buf });
        }
      } catch (_) {}
    }
  }
}

sendBtn.addEventListener("click", sendMessage);
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

clearBtn.addEventListener("click", () => {
  history = [];
  chatEl.innerHTML = "";
});

exportBtn.addEventListener("click", async () => {
  const res = await fetch("/api/export", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ history }),
  });
  const { filename } = await res.json();
  const a = document.createElement("a");
  a.href = "/" + filename;
  a.download = filename;
  a.click();
});

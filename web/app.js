const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
let state = { stats: {}, settings: {}, tasks: [], cards: [], feeds: [], report: {}, agent: {} };
let graphData = { nodes: [], edges: [] };
let dailyItems = [];
const taskCache = new Map();
let activeReviewTask = null;
let lastAgentExchange = null;
let reviewConversation = [];
let inspirationConversation = [];
try { inspirationConversation = JSON.parse(localStorage.getItem("garden.inspirationConversation") || "[]"); } catch (_) { inspirationConversation = []; }
let inspirationSessionId = localStorage.getItem("garden.inspirationSessionId") || crypto.randomUUID();
let latestInspiration = null;
let agentConversation = [];
try { agentConversation = JSON.parse(localStorage.getItem("garden.agentConversation") || "[]"); } catch (_) { agentConversation = []; }
const AGENT_HISTORY_LIMIT = 20;
const AGENT_VISIBLE_MESSAGES = 8;
let agentHistoryExpanded = false;
let agentSessionId = localStorage.getItem("garden.agentSessionId") || crypto.randomUUID();
localStorage.setItem("garden.agentSessionId", agentSessionId);
let showProposedCrossLinks = localStorage.getItem("garden.showProposedCrossLinks") === "1";
let wechatPreview = null;
let officialAccounts = [];
let officialArticles = [];
const officialArticlePreviews = new Map();
let readingArticle = null;
let readingConversation = [];
let readingSessionId = crypto.randomUUID();

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || "操作没有完成");
  return data;
}

function toast(message, error = false) {
  const node = $("#toast"); node.textContent = message; node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.className = "toast", 3000);
}
function busy(show, message = "知识园丁正在松土……") { $("#busy span").textContent = message; $("#busy").classList.toggle("show", show); }
function escapeHTML(value = "") { return String(value).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c])); }
function safeURL(value = "") { try { const url = new URL(value, location.origin); return ["http:", "https:"].includes(url.protocol) ? url.href : "#"; } catch { return "#"; } }
function excerpt(value = "", length = 85) { const clean = value.replace(/[#*`\n]/g, " ").replace(/\s+/g, " "); return clean.length > length ? clean.slice(0, length) + "…" : clean; }
function withoutEmbeddedMaterial(value = "") {
  const clean = String(value).split(/<frontier_material>/i)[0].trim();
  return clean || "开始一篇文章的互动导读";
}
function formatGardenerMarkdown(value = "") {
  let text=String(value).trim();
  const headings={
    "结论":"先说结论","机制":"为什么","边界":"成立边界",
    "适用条件":"适用条件","证据缺口":"目前还缺什么证据","反例":"反例与限制"
  };
  const labels=Object.keys(headings).join("|");
  text=text.replace(new RegExp(`(^|[。！？；]\\s*)(${labels})：\\s*`,"g"),(match,prefix,label)=>{
    const before=prefix.trim();
    return `${before}${before?"\n\n":""}### ${headings[label]}\n\n`;
  });
  const blocks=text.split(/\n{2,}/).flatMap(block=>{
    const clean=block.trim();
    if(!clean||/^(#{1,4}|[-*>])\s/.test(clean)||clean.length<240)return [clean];
    const sentences=clean.match(/[^。！？]+[。！？]?/g)?.map(item=>item.trim()).filter(Boolean)||[clean];
    if(sentences.length<5)return [clean];
    const paragraphs=[];
    for(let index=0;index<sentences.length;index+=3)paragraphs.push(sentences.slice(index,index+3).join(""));
    return paragraphs;
  });
  return blocks.filter(Boolean).join("\n\n");
}
agentConversation = agentConversation.slice(-AGENT_HISTORY_LIMIT).map(message => (
  message?.role === "user" ? {...message, content:withoutEmbeddedMaterial(message.content)} : message
));
function renderMarkdown(value = "") {
  const inline = text => escapeHTML(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g, '<span class="wiki-link">$1</span>');
  return String(value).split(/\r?\n/).map(line => {
    const clean = line.trim();
    if (!clean || clean.startsWith("<!--") || clean === "---") return "";
    if (/^#\s+/.test(clean)) return "";
    const heading = clean.match(/^(#{2,4})\s+(.+)$/);
    if (heading) return `<h4>${inline(heading[2])}</h4>`;
    if (clean.startsWith("> ")) return `<div class="md-quote">${inline(clean.slice(2))}</div>`;
    if (clean.startsWith("- ")) return `<div class="md-list">• ${inline(clean.slice(2))}</div>`;
    return `<p>${inline(clean)}</p>`;
  }).join("");
}

function renderPlannerAudit(result = {}) {
  const plan=result.planner||{};
  if(!plan.goal)return "";
  const steps=(result.agent_trace||[]).map(item=>item.summary).filter(Boolean).slice(0,14);
  const modality=plan.primary_modality==="text_visual"?`文字 + ${plan.visual_kind||"图解"}`:"文字";
  return `<details class="planner-audit"><summary>园丁这次怎样组织回答 · ${escapeHTML(modality)}</summary><div class="planner-audit-body"><p><b>目标</b>${escapeHTML(plan.goal)}</p><p><b>表达选择</b>${escapeHTML(plan.modality_reason||"按当前任务选择最小充分表达。")}</p>${plan.visual_request?`<p><b>发给 DeepDiagram 的任务</b>${escapeHTML(plan.visual_request)}</p>`:""}${steps.length?`<ol>${steps.map(item=>`<li>${escapeHTML(item)}</li>`).join("")}</ol>`:""}${result.revision_count?`<small>Reflector 已执行 ${escapeHTML(result.revision_count)} 次定向返工。</small>`:""}</div></details>`;
}

function renderAgentVisualization(spec = {}) {
  if(spec.status!=="ready"||!Array.isArray(spec.nodes)||spec.nodes.length<2)return "";
  const nodes=spec.nodes.slice(0,18), edges=(spec.edges||[]).slice(0,28);
  const byId=new Map(nodes.map(node=>[String(node.id),node]));
  const incoming=new Map(nodes.map(node=>[String(node.id),0]));
  edges.forEach(edge=>{if(byId.has(String(edge.source))&&byId.has(String(edge.target)))incoming.set(String(edge.target),(incoming.get(String(edge.target))||0)+1)});
  const roots=nodes.filter(node=>(incoming.get(String(node.id))||0)===0).map(node=>String(node.id));
  const level=new Map(roots.map(id=>[id,0]));
  for(let pass=0;pass<nodes.length;pass++)edges.forEach(edge=>{const a=String(edge.source),b=String(edge.target);if(level.has(a))level.set(b,Math.max(level.get(b)||0,level.get(a)+1))});
  nodes.forEach(node=>{if(!level.has(String(node.id)))level.set(String(node.id),0)});
  const groups=new Map();nodes.forEach(node=>{const key=level.get(String(node.id))||0;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(node)});
  const maxLevel=Math.max(...groups.keys(),0), boxW=154,boxH=58,xGap=54,yGap=22;
  const width=Math.max(720,(maxLevel+1)*(boxW+xGap)+70), maxGroup=Math.max(...[...groups.values()].map(items=>items.length),1);
  const height=Math.max(180,maxGroup*(boxH+yGap)+70), positions=new Map();
  [...groups.entries()].sort((a,b)=>a[0]-b[0]).forEach(([column,items])=>{const total=items.length*boxH+(items.length-1)*yGap;items.forEach((node,index)=>positions.set(String(node.id),{x:35+column*(boxW+xGap),y:(height-total)/2+index*(boxH+yGap)}))});
  const marker=`garden-arrow-${Math.random().toString(36).slice(2)}`;
  const lines=edges.map(edge=>{const a=positions.get(String(edge.source)),b=positions.get(String(edge.target));if(!a||!b)return "";const x1=a.x+boxW,y1=a.y+boxH/2,x2=b.x,y2=b.y+boxH/2,mid=(x1+x2)/2;return `<g><path d="M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}" marker-end="url(#${marker})"/><text x="${mid}" y="${(y1+y2)/2-5}" text-anchor="middle">${escapeHTML(edge.label||"")}</text></g>`}).join("");
  const boxes=nodes.map(node=>{const pos=positions.get(String(node.id)),sources=(node.evidence_ids||[]).join(" · ");return `<g class="agent-diagram-node"><rect x="${pos.x}" y="${pos.y}" width="${boxW}" height="${boxH}" rx="13"/><foreignObject x="${pos.x+8}" y="${pos.y+7}" width="${boxW-16}" height="${boxH-12}"><div xmlns="http://www.w3.org/1999/xhtml"><b>${escapeHTML(node.label)}</b>${sources?`<small>${escapeHTML(sources)}</small>`:""}</div></foreignObject></g>`}).join("");
  const kindLabels={mindmap:"知识层级图",flowchart:"机制流程图",timeline:"发展时间线",comparison:"比较关系图",concept:"空间概念图"};
  const providerLabels={"deepdiagram-full":"完整 DeepDiagram","local-deterministic-adapter":"本地图解回退","deepdiagram-compatible":"DeepDiagram 兼容适配层"};
  return `<section class="agent-visualization"><header><div><span>${escapeHTML(kindLabels[spec.kind]||"知识图解")}</span><h4>${escapeHTML(spec.title||"知识图解")}</h4></div><small>${escapeHTML(providerLabels[spec.provider]||spec.provider||"安全图解")}</small></header>${spec.design_concept?`<p>${escapeHTML(spec.design_concept)}</p>`:""}<div class="agent-diagram-scroll"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHTML(spec.title||"知识图解")}"><defs><marker id="${marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker></defs><g class="agent-diagram-edges">${lines}</g>${boxes}</svg></div>${spec.warning?`<small class="diagram-warning">${escapeHTML(spec.warning)}</small>`:""}</section>`;
}

function switchView(name) {
  $$(".nav").forEach(n => n.classList.toggle("active", n.dataset.view === name));
  $$(".view").forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  const titles = {home:"早上好，园丁",frontier:"前沿雷达",garden:"我的知识树",capture:"灵感温室",wechat:"微信苗圃",settings:"花园设置"};
  $("#page-title").textContent = titles[name];
  if (name === "garden") loadGarden();
  if (name === "frontier") loadFrontierNotes();
  if (name === "wechat") loadWechatCandidates();
  scrollTo({top:0,behavior:"smooth"});
}

async function bootstrap() {
  state = await api("/api/bootstrap");
  renderAll();
  loadDaily();
}

function renderAll() {
  const s = state.stats;
  $("#m-notes").textContent = s.notes || 0; $("#m-concepts").textContent = s.concepts || 0;
  $("#m-completed").textContent = s.completed_tasks || 0; $("#m-links").textContent = s.links || 0;
  $("#side-level").textContent = `Lv.${s.level || 1}`; $("#side-xp").textContent = `${s.level_progress || 0} / 100 XP`;
  $("#side-progress").style.width = `${s.level_progress || 0}%`;
  $("#sync-label").textContent = state.settings.vault_path ? "Obsidian 已连接" : "等待连接 Obsidian";
  $("#sync-dot").style.background = state.settings.vault_path ? "#8ebd7c" : "#d9a843";
  $("#vault-path").value = state.settings.vault_path || ""; $("#learning-level").value = state.settings.learning_level || "本科入门";
  $("#interests").value = (state.settings.interests || []).join("、");
  $("#textbook-directory").value = state.settings.textbook_directory || "";
  const pendingClassification=(state.settings.classification_queue||[]).length;
  if($("#rebuild-links"))$("#rebuild-links").textContent=pendingClassification?`复核待分类新知（${pendingClassification}）`:"重新整理结构与连接";
  const llmOn = state.settings.llm_enabled;
  const llmState = state.settings.llm_status || (llmOn ? "connected" : "offline");
  $("#llm-status").textContent = state.settings.llm_message || (llmOn ? "已启用兼容大模型，将按你的水平生成讲解。" : "当前使用离线规则；配置理解 API 可启用深度讲解。");
  const badgeLabels = { connected: "已连接", checking: "验证中", invalid_key: "密钥无效", limited: "额度受限", error: "连接失败", offline: "离线模式" };
  $("#llm-badge").textContent = badgeLabels[llmState] || "离线模式"; $("#llm-badge").classList.toggle("on", llmOn);
  renderWechatStatus(state.settings.wechat || {});
  renderTasks("#task-list", state.tasks.slice(0, 4)); renderCards(); renderReport(); renderFeeds();
  renderAgent();
}

function renderWechatStatus(status) {
  const online = Boolean(status.authorized);
  const badge = $("#wechat-main-badge");
  if (badge) { badge.textContent = online ? "已授权" : status.service_online ? "待 Token" : "未连接"; badge.classList.toggle("on", online); }
  if ($("#tracememo-base-url")) $("#tracememo-base-url").value = status.base_url || "http://127.0.0.1:6131";
  const detail = status.message || "TraceMemo 尚未连接。";
  if ($("#wechat-connection-detail")) $("#wechat-connection-detail").innerHTML = `<b>${escapeHTML(detail)}</b><small>Token：${status.token_saved ? "已由 Windows DPAPI 保存" : status.token_configured ? "仅本次进程使用" : "未配置"} · 数据库与密钥不会进入知识花园</small>`;
  if ($("#wechat-settings-status")) $("#wechat-settings-status").textContent = detail;
}

function candidateArray(value) {
  if (Array.isArray(value)) return value;
  if (!value || typeof value !== "object") return [];
  for (const key of ["chats","conversations","contacts","items","data","recentChats","recent_chats"]) if (Array.isArray(value[key])) return value[key];
  return [];
}

function recentLabel(item) {
  const names = [item.display_name, item.m_nsNickName, item.displayName, item.wechatNickname, item.nickname, item.name, item.remark, item.account_name, item.talker];
  const humanName = names.map(value => String(value || "").trim()).find(value => value && !/^(?:gh_[a-z0-9]+(?:@app)?|wxid_[a-z0-9_]+|[a-f0-9]{24,64})$/i.test(value));
  return humanName || "暂未识别真实名称";
}

async function loadOfficialAccounts() {
  busy(true, "正在从本地微信识别公众号……");
  try {
    const data = await api("/api/wechat/official-accounts");
    officialAccounts = candidateArray(data.result);
    const select = $("#wechat-official-account");
    select.innerHTML = '<option value="">选择公众号</option>' + officialAccounts.map(item => {
      const name = recentLabel(item);
      return `<option value="${escapeHTML(name)}">${escapeHTML(name)}</option>`;
    }).join("");
    const unresolved = Number(data.result?.unresolved_count || 0);
    $("#wechat-official-summary").innerHTML = `<b>已识别 ${officialAccounts.length} 个公众号</b><small>这里只展示已经识别出真实昵称的公众号，不扫描普通联系人和群聊。${unresolved ? `另有 ${unresolved} 个账号暂未取得真实名称，已隐藏内部标识。` : ""}</small>`;
  } catch (err) { toast(err.message, true); } finally { busy(false); }
}

function quickArticleMeta(article) {
  const text = `${article.title || ""} ${article.description || ""}`.toLowerCase();
  const rules = [
    ["心理学", ["心理","认知","情绪","人格","行为","神经"]],
    ["历史与文化", ["历史","宫廷","古代","王朝","文物","考古","传统","文化"]],
    ["文学与语言", ["文学","诗歌","小说","语言","叙事","修辞","翻译"]],
    ["哲学与思想", ["哲学","认识论","伦理","存在","意识","思想"]],
    ["经济与商业", ["经济","金融","市场","企业","商业","投资","产业"]],
    ["计算机与人工智能", ["人工智能","ai","算法","模型","机器学习","神经网络","大模型"]],
    ["电子信息与工程", ["电路","电子","芯片","通信","控制","机器人","嵌入式"]],
    ["生命科学与医学", ["生物","医学","疾病","临床","细胞","基因","药物","健康"]],
    ["艺术与审美", ["艺术","审美","绘画","音乐","舞蹈","电影","建筑","设计"]],
  ];
  const ranked = rules.map(([domain, terms]) => ({domain, terms:terms.filter(term => text.includes(term)), score:terms.reduce((n,term)=>n+(text.includes(term)?1:0),0)})).filter(item=>item.score).sort((a,b)=>b.score-a.score);
  return {domain:ranked[0]?.domain || "待读正文判断", knowledge:(ranked[0]?.terms || []).slice(0,4)};
}

function updateOfficialArticlePreview(index, preview) {
  const article = officialArticles[index]?.article || {};
  article.preview = preview;
  officialArticlePreviews.set(article.url, preview);
  const domain = $(`#official-domain-${index}`), reading = $(`#official-reading-${index}`), summary = $(`#official-summary-${index}`), knowledge = $(`#official-knowledge-${index}`), status = $(`#official-status-${index}`);
  if (domain) domain.textContent = `领域 · ${preview.domain || "待读正文判断"}`;
  if (reading) reading.textContent = preview.reading_minutes ? `预计阅读 · ${preview.reading_minutes} 分钟` : "预计阅读 · 待读正文";
  if (summary) summary.innerHTML = `<b>文章摘要</b>${escapeHTML(preview.summary || article.description || "暂未取得摘要")}`;
  if (knowledge) knowledge.innerHTML = (preview.knowledge || []).length ? `<b>涉及知识</b>${preview.knowledge.map(item=>`<span>${escapeHTML(item)}</span>`).join("")}` : "<b>涉及知识</b><span>正文证据不足，暂不判断</span>";
  if (status) status.textContent = preview.summary_source ? `${preview.summary_source} · ${preview.classification_status || "初步识别"}` : "正文初步识别";
}

async function hydrateOfficialArticlePreviews() {
  for (let index = 0; index < Math.min(officialArticles.length, 12); index += 1) {
    const article = officialArticles[index]?.article || {};
    if (!article.url) continue;
    const cached = officialArticlePreviews.get(article.url);
    if (cached) { updateOfficialArticlePreview(index, cached); continue; }
    try {
      const data = await api("/api/wechat/article/preview", {method:"POST", body:JSON.stringify({url:article.url,title:article.title||"",description:article.description||""})});
      updateOfficialArticlePreview(index, data.preview || {});
    } catch (_) {
      const reading = $(`#official-reading-${index}`), status = $(`#official-status-${index}`);
      if (reading) reading.textContent = "预计阅读 · 打开原文后计算";
      if (status) status.textContent = "正文回源受限 · 当前标签仅依据文章卡片";
    }
  }
}

function renderOfficialArticles(result) {
  const accountName = result.account_name || result.contact?.m_nsNickName || result.talker || "未命名公众号";
  result = {...result, account_name: accountName, talker: accountName};
  officialArticles = (result.articles || []).map(message => {
    const article = message.article || {};
    const articleAccount = article.account_name || message.account_name || accountName;
    return {...message, account_name: articleAccount, sender: articleAccount, article: {
      ...article, original_publisher: article.publisher || "", account_name: articleAccount,
      publisher: articleAccount || article.publisher || "",
    }};
  });
  const host = $("#wechat-article-inbox");
  host.innerHTML = `<article class="panel official-inbox-panel"><div class="panel-head"><div><span class="kicker">ARTICLE INBOX</span><h3>${escapeHTML(result.talker)} · ${officialArticles.length} 篇</h3><small>最近 ${escapeHTML(result.days)} 天${result.truncated ? " · 消息较多，已限制本轮读取量" : ""}</small></div></div><div class="wechat-message-list official-article-list">${officialArticles.length ? officialArticles.map((message,index)=>{const article=message.article||{},quick=quickArticleMeta(article),cached=officialArticlePreviews.get(article.url)||{};const preview={...quick,...cached,knowledge:cached.knowledge||quick.knowledge};return `<article class="wechat-message official-article-card"><div class="official-article-main"><div class="official-article-source"><small>${escapeHTML(article.publisher||message.sender||result.talker)} · ${escapeHTML(message.sent_at||"")}</small><small id="official-status-${index}">${cached.summary_source ? `${escapeHTML(cached.summary_source)} · ${escapeHTML(cached.classification_status||"初步识别")}` : "正在回源读取正文信息…"}</small></div><h4>${escapeHTML(article.title||"未命名文章")}</h4><div class="official-article-meta"><span id="official-domain-${index}">领域 · ${escapeHTML(preview.domain)}</span><span id="official-reading-${index}">${preview.reading_minutes ? `预计阅读 · ${preview.reading_minutes} 分钟` : "预计阅读 · 正在计算"}</span></div><p class="official-article-summary" id="official-summary-${index}"><b>文章摘要</b>${escapeHTML(preview.summary||article.description||"正在读取正文，以生成可判断内容价值的摘要…")}</p><div class="official-knowledge" id="official-knowledge-${index}"><b>涉及知识</b>${preview.knowledge.length ? preview.knowledge.map(item=>`<span>${escapeHTML(item)}</span>`).join("") : "<span>正在从正文提取</span>"}</div><div class="button-row official-article-actions"><a class="secondary" href="${escapeHTML(safeURL(article.url))}" target="_blank" rel="noopener noreferrer">阅读原文</a><button class="primary official-guide" data-index="${index}" type="button">读取正文并互动导读</button><button class="text-btn official-queue" data-index="${index}" type="button">加入待审核区</button></div></div></article>`}).join("") : '<div class="empty">这个公众号在所选时间内没有返回文章卡片，可以扩大到最近 90 天。</div>'}</div></article>`;
  $$(".official-guide").forEach(button => button.onclick = () => startOfficialReading(Number(button.dataset.index)));
  $$(".official-queue").forEach(button => button.onclick = () => queueOfficialArticle(Number(button.dataset.index), result));
  void hydrateOfficialArticlePreviews();
}

async function startOfficialReading(index) {
  const message = officialArticles[index];
  const article = message?.article || {};
  if (!article.url) return;
  readingArticle = {title:article.title||"这篇公众号文章",url:article.url,publisher:article.account_name||article.publisher||message.account_name||message.sender||"",sentAt:message.sent_at||"",text:"",scope:"open_fulltext"};
  readingConversation = [];
  readingSessionId = crypto.randomUUID();
  $("#reading-title").textContent = readingArticle.title;
  $("#reading-byline").textContent = [readingArticle.publisher, readingArticle.sentAt].filter(Boolean).join(" · ");
  $("#reading-original-link").href = safeURL(readingArticle.url);
  $("#reading-source-status").textContent = "正在从公众号原文提取正文……";
  $("#reading-source").innerHTML = '<div class="empty">正在准备不打断聊天的独立原文窗口……</div>';
  $("#reading-conversation").innerHTML = '<div class="empty">正文准备好后，园丁会先邀请你做读前预测。</div>';
  $("#reading-workspace").classList.add("show");
  $("#reading-workspace").setAttribute("aria-hidden", "false");
  let materialText = article.description || "";
  let scope = "abstract";
  try {
    const data = await api("/api/wechat/article/read", {method:"POST", body:JSON.stringify({url:article.url})});
    materialText = data.text || materialText;
    scope = data.access_scope || "open_fulltext";
    $("#reading-source-status").textContent = `已提取正文 · ${Math.max(1,Math.ceil(materialText.replace(/\s/g,"").length/450))} 分钟`;
  } catch (err) {
    $("#reading-source-status").textContent = "正文回源受限 · 当前仅显示文章卡片摘要";
    toast(`正文回源受限，将先依据文章卡片摘要导读：${err.message}`, true);
  }
  readingArticle.text = String(materialText).replace(/<\/frontier_material>/gi, "");
  readingArticle.scope = scope;
  const paragraphs = readingArticle.text.split(/\n+/).map(part=>part.trim()).filter(Boolean);
  $("#reading-source").innerHTML = paragraphs.length ? paragraphs.map(part=>`<p>${escapeHTML(part)}</p>`).join("") : '<div class="empty">没有取得可阅读文本，请使用“浏览器打开原文”。</div>';
  await sendReadingQuestion("请先不要概括全文，只根据这篇文章的标题和开头向我提出一个具体的读前预测问题，等待我回答后再继续。", false);
}

function readingMaterialContext() {
  if (!readingArticle) return "";
  const body = readingArticle.text.slice(0, 18000);
  return `<frontier_material>\ntitle: ${readingArticle.title}\nurl: ${readingArticle.url}\nauthors: ${readingArticle.publisher}\nvenue: 微信公众号\naccess_scope: ${readingArticle.scope}\nbody:\n${body}\n</frontier_material>`;
}

function renderReadingConversation() {
  const host = $("#reading-conversation");
  if (!readingConversation.length) { host.innerHTML = '<div class="empty">园丁正在准备第一个读前问题……</div>'; return; }
  host.innerHTML = readingConversation.map(message=>{
    if (message.role === "user") return `<div class="chat-turn user-turn"><small>你</small><p>${escapeHTML(message.content)}</p></div>`;
    const result = message.result || {};
    const prompts = (result.discussion_prompts||[]).slice(0,3).map(prompt=>`<button class="secondary reading-prompt" type="button" data-prompt="${escapeHTML(prompt)}">${escapeHTML(prompt)}</button>`).join("");
    return `<div class="chat-turn assistant-turn"><small>导读园丁</small>${renderMarkdown(message.content)}${result.followup?`<div class="review-followup"><b>接着可以想</b><p>${escapeHTML(result.followup)}</p>${prompts}</div>`:""}</div>`;
  }).join("");
  $$(".reading-prompt").forEach(button=>button.onclick=()=>{$("#reading-question").value=button.dataset.prompt;$("#reading-question").focus()});
  host.scrollTop = host.scrollHeight;
}

async function sendReadingQuestion(question, showUser = true) {
  question = String(question||"").trim();
  if (!question || !readingArticle) return;
  const prior = readingConversation.map(item=>({role:item.role,content:item.content})).slice(-8);
  if (showUser) readingConversation.push({role:"user",content:question});
  renderReadingConversation();
  const submit = $("#reading-question-form button");
  submit.disabled = true; submit.textContent = "园丁正在阅读…";
  try {
    const history = [{role:"user",content:readingMaterialContext()},...prior];
    const data = await api("/api/agent/ask", {method:"POST",body:JSON.stringify({question,history,session_id:readingSessionId})});
    const result = data.result;
    readingSessionId = result.session_id || readingSessionId;
    readingConversation.push({role:"assistant",content:result.answer,result});
    renderReadingConversation();
  } catch (err) {
    if (showUser) readingConversation.pop();
    renderReadingConversation();
    toast(err.message,true);
  } finally {
    submit.disabled = false; submit.textContent = "继续导读";
  }
}

async function queueOfficialArticle(index, result) {
  const message = officialArticles[index];
  const article = message?.article || {};
  if (!message) return;
  busy(true, "正在把公众号文章放入 L1 待审核区……");
  try {
    await api("/api/wechat/candidates", {method:"POST", body:JSON.stringify({
      title: article.title || `${result.account_name||result.talker} · 公众号文章`,
      talker: result.account_name||result.talker, time: message.sent_at || "",
      contact: result.contact || {}, query: {source:"official_account", article:{
        title:article.title||"", url:article.url||"", account_name:article.account_name||result.account_name||"", publisher:article.publisher||article.account_name||message.sender||result.account_name||result.talker,
        description:article.description||"", sent_at:message.sent_at||"", preview:article.preview||null
      }}, messages:[message],
    })});
    toast("已加入待审核区；尚未写入 Obsidian 或更新掌握度");
    await loadWechatCandidates();
  } catch (err) { toast(err.message, true); } finally { busy(false); }
}

async function loadWechatCandidates() {
  try {
    const data = await api("/api/wechat/candidates");
    const candidates = data.candidates || [];
    $("#wechat-candidates").innerHTML = candidates.length ? candidates.map(item => `<article class="wechat-candidate ${escapeHTML(item.status)}"><div><span>${item.status === "pending" ? "待审核" : item.status === "accepted" ? "已沉淀" : "已拒绝"}</span><h4>${escapeHTML(item.title)}</h4><p>${escapeHTML(item.talker || "未知会话")} · ${escapeHTML(item.time_range || "未限定时间")} · ${item.message_count} 条已选片段</p>${item.raw_path ? `<small>Obsidian：${escapeHTML(item.raw_path)}</small>` : ""}</div>${item.status === "pending" ? `<div class="candidate-actions"><button class="primary" onclick="reviewWechatCandidate('${item.candidate_id}',true)">确认沉淀</button><button class="secondary" onclick="reviewWechatCandidate('${item.candidate_id}',false)">拒绝</button></div>` : ""}</article>`).join("") : '<div class="empty">还没有微信候选。先按会话读取，并选择真正有价值的片段。</div>';
  } catch (err) { $("#wechat-candidates").innerHTML = `<div class="empty">候选暂时无法读取：${escapeHTML(err.message)}</div>`; }
}

function renderWechatPreview(preview) {
  wechatPreview = preview;
  const allMessages = preview.messages || [];
  const messages = allMessages.filter(message => !message.is_system);
  wechatPreview = {...preview, messages};
  const host = $("#wechat-preview");
  host.innerHTML = `<article class="panel"><div class="panel-head"><div><span class="kicker">LOCAL PREVIEW</span><h3>${escapeHTML(preview.talker)} · 有效消息 ${messages.length} 条</h3><small>已过滤 ${allMessages.length-messages.length} 条入群、撤回等系统通知。TraceMemo 报告总数 ${preview.count ?? allMessages.length}${preview.truncated ? "；为保护上下文已截取前 300 条" : ""}</small></div><label class="check-row"><input id="wechat-select-all" type="checkbox" checked> 全选</label></div><div class="wechat-message-list">${messages.length ? messages.map((m,index)=>`<label class="wechat-message"><input class="wechat-message-check" type="checkbox" value="${index}" checked><div><small>${escapeHTML(m.sender || "未知发送者")} · ${escapeHTML(m.sent_at || "时间未知")}</small><p>${escapeHTML(m.content)}</p>${m.article?.url?`<a href="${escapeHTML(safeURL(m.article.url))}" target="_blank" rel="noopener noreferrer">打开文章原文</a>`:""}</div></label>`).join("") : '<div class="empty">过滤系统通知后没有可用内容，请调整会话或日期。</div>'}</div>${messages.length ? '<div class="preview-foot"><p>创建候选只记录为 L1，不会更新掌握度、长期画像或知识图谱。</p><button id="wechat-create-candidate" class="primary">把选中片段放入待审核区</button></div>' : ""}</article>`;
  $("#wechat-select-all")?.addEventListener("change", e => $$(".wechat-message-check").forEach(box => box.checked = e.target.checked));
  $("#wechat-create-candidate")?.addEventListener("click", createWechatCandidate);
}

async function createWechatCandidate() {
  if (!wechatPreview) return;
  const selected = $$(".wechat-message-check:checked").map(box => wechatPreview.messages[Number(box.value)]).filter(Boolean);
  if (!selected.length) { toast("请至少选择一条消息", true); return; }
  busy(true, "正在建立可追溯的 L1 微信候选……");
  try {
    await api("/api/wechat/candidates", {method:"POST", body:JSON.stringify({
      title: $("#wechat-candidate-title").value.trim() || `${wechatPreview.talker} · 微信讨论`,
      talker: wechatPreview.talker, time: wechatPreview.time,
      contact: wechatPreview.contact, query: wechatPreview.query, messages: selected,
    })});
    toast(`已建立 L1 候选：${selected.length} 条片段；尚未写入 Obsidian`);
    await loadWechatCandidates();
    $("#wechat-candidates").scrollIntoView({behavior:"smooth"});
  } catch (err) { toast(err.message, true); } finally { busy(false); }
}

async function reviewWechatCandidate(candidateId, accepted) {
  busy(true, accepted ? "正在按你的确认写入 raw、执行 Ingest 并更新知识图谱……" : "正在记录拒绝决定……");
  try {
    const data = await api("/api/wechat/candidates/review", {method:"POST", body:JSON.stringify({candidate_id:candidateId, accepted})});
    toast(accepted ? `已沉淀到 ${data.candidate.raw_path}，并完成知识编译` : "已拒绝；内容不会进入 Obsidian 或知识图谱");
    await loadWechatCandidates(); await bootstrap();
  } catch (err) { toast(err.message, true); } finally { busy(false); }
}
window.reviewWechatCandidate = reviewWechatCandidate;

function renderAgent() {
  const agent=state.agent||{}; const task=agent.task;
  $("#agent-status").textContent=agent.status||"巡视中";
  $("#agent-observation").textContent=agent.observation||"园丁正在观察知识库的变化。";
  $("#agent-action").textContent=agent.action||"";
  $("#agent-question").textContent=agent.question||"你现在最想真正理解什么？";
  const button=$("#agent-review");
  if(task){taskCache.set(task.id,task);button.style.display="inline-block";button.onclick=()=>openReview(task.id)}
  else{button.style.display="none"}
}

function renderTasks(selector, tasks) {
  tasks.forEach(task => taskCache.set(task.id, task));
  const host = $(selector); if (!tasks.length) { host.innerHTML = '<div class="empty">今天没有待办。去种下一颗新种子吧。</div>'; return; }
  host.innerHTML = tasks.map(t => {const prompt=t.payload?.question||((t.payload?.questions||[])[0])||"";return `<div class="task ${t.status === "done" ? "done" : ""}"><span class="task-icon">${{quiz:"?",recall:"↺",socratic:"✦",frontier:"⌁"}[t.task_type] || "•"}</span><div class="task-copy"><b>${escapeHTML(t.title)}</b><small>${t.xp} XP · ${t.task_type === "quiz" ? "小测验" : t.task_type === "socratic" ? "苏格拉底追问" : t.task_type === "frontier" ? "前沿待嫁接" : "主动回忆"}</small>${prompt?`<p class="task-prompt">${escapeHTML(prompt)}</p>`:""}</div>${t.status === "done" ? '<span>✓</span>' : `<button onclick="openReview(${t.id})">${t.task_type === "frontier" ? "学习" : "回答"}</button>`}</div>`}).join("");
}

function renderCards() {
  const cards = state.cards || []; const host = $("#recent-cards");
  host.innerHTML = cards.length ? cards.slice(0,3).map(c => `<div class="bloom-card"><small>✿ 教材 × 前沿</small><h4>${escapeHTML(c.concept)}</h4><p>${escapeHTML(excerpt(c.explanation))}</p></div>`).join("") : '<div class="empty">第一朵花还在等待。分析一篇前沿材料，让它绽放。</div>';
}

function renderReport() {
  const r = state.report || {}; $("#report-title").textContent = r.title || "学习周报"; $("#report-insight").textContent = r.insight || "你的花园正在准备发芽。";
  $("#report-concepts").innerHTML = (r.new_concepts || []).map(c => `<span class="chip">${escapeHTML(c)}</span>`).join("") || '<span class="chip">等待新概念</span>';
  $("#report-done").textContent = `本周完成 ${r.completed || 0} 项`;
  $("#hero-text").textContent = r.insight || $("#hero-text").textContent;
}

function renderFeeds() {
  const host = $("#feed-list"); const feeds = state.feeds || [];
  host.innerHTML = feeds.length ? feeds.map(f => `<div class="feed-item"><div><b>${escapeHTML(f.name)}</b><small>${escapeHTML(f.url)}</small></div><span class="chip">追踪中</span></div>`).join("") : '<div class="empty">添加技术博主的 RSS 地址，更新就会进入花园。</div>';
}

async function loadDaily(force=false){try{const data=await api(`/api/daily${force?"?refresh=1":""}`);dailyItems=data.items||[];$("#daily-profile").textContent=data.interests?.length?`垂类画像：${data.level} · ${data.interests.join(" / ")}`:"还没有设置兴趣画像";const notice=data.notice?`<div class="source-notice">${escapeHTML(data.notice)}</div>`:"";$("#daily-digest").innerHTML=notice+(dailyItems.length?dailyItems.map((item,index)=>{const g=item.reading_guide||{},scores=item.scores||{},links=(item.connections||[]).map(c=>`《${escapeHTML(c.title)}》`).join("、");return `<article class="daily-card ${item.read?"is-read":""}"><span>${escapeHTML(item.interest||"前沿推荐")} · ${escapeHTML(item.year||"")} ${item.read?"· 已读":""}</span><h4><a href="${escapeHTML(safeURL(item.url))}" target="_blank" rel="noopener noreferrer">${escapeHTML(item.title)}</a></h4><p>${escapeHTML(item.why)}</p>${links?`<small>知识树连接：${links}</small>`:""}<div class="frontier-scores"><span>领域 ${Math.round((scores.domain_match||0)*100)}%</span><span>衔接 ${Math.round((scores.knowledge_connection||0)*100)}%</span><span>来源 ${Math.round((scores.source_quality||0)*100)}%</span><span>新鲜 ${Math.round((scores.freshness||0)*100)}%</span></div><details class="reading-guide"><summary>打开互动导读</summary><div><b>读前 · 先预测</b><p>${escapeHTML(g.before_reading||"")}</p><b>方向提示</b><p>${escapeHTML(g.orientation||"")}</p><b>读中检查点</b><ol>${(g.checkpoints||[]).map(q=>`<li>${escapeHTML(q)}</li>`).join("")}</ol><b>读后 · 带走什么</b><p>${escapeHTML(g.after_reading||"")}</p></div></details><div class="daily-actions"><button class="primary" onclick="askDaily(${index})">开始互动导读</button><button class="secondary" onclick="markDailyRead(${index})" ${item.read?"disabled":""}>${item.read?"已标记阅读":"标记已读"}</button><button class="text-btn" onclick="saveDaily(${index})">确认加入知识库</button></div></article>`}).join(""):`<div class="empty">${escapeHTML(data.message||"今天暂时没有推荐。")}</div>`)}catch(err){$("#daily-digest").innerHTML=`<div class="empty">每日推送暂时不可用：${escapeHTML(err.message)}</div>`}}
function askDaily(index){const item=dailyItems[index];if(!item)return;switchView("home");const material=`<frontier_material>\ntitle: ${item.title||""}\nurl: ${item.url||""}\npdf_url: ${item.pdf_url||""}\nauthors: ${(item.authors||[]).join("; ")}\nvenue: ${item.venue||""}\nyear: ${item.year||""}\naccess_scope: ${item.abstract?"abstract":"metadata_only"}\nabstract:\n${item.abstract||""}\n</frontier_material>`;$("#agent-ask-input").value=`${item.prompt}\n\n${material}`;$("#agent-ask-input").focus();toast("巡视材料及来源已带入输入框；你可以修改导读要求后再发送")}
async function saveDaily(index){const item=dailyItems[index];if(!item)return;busy(true,"正在把推荐资料种入知识库并建立学习入口……");try{await api("/api/daily/save",{method:"POST",body:JSON.stringify(item)});toast("推荐资料已加入知识库，并进入知识编译流程");await bootstrap();await loadGarden()}catch(err){toast(err.message,true)}finally{busy(false)}}
async function markDailyRead(index){const item=dailyItems[index];if(!item)return;try{await api("/api/daily/read",{method:"POST",body:JSON.stringify({url:item.url,title:item.title})});item.read=true;toast("已记录阅读，之后的推荐会避开重复材料 +2 XP");await loadDaily()}catch(err){toast(err.message,true)}}
window.askDaily=askDaily;window.saveDaily=saveDaily;window.markDailyRead=markDailyRead;

async function completeTask(id) {
  try { const result = await api("/api/tasks/complete", {method:"POST",body:JSON.stringify({id})}); toast("完成浇灌，经验值已生长 +"); await bootstrap(); if ($("#view-garden").classList.contains("active")) loadGarden(); } catch(e) { toast(e.message,true); }
}
window.completeTask = completeTask;

function openReview(id) {
  const task=taskCache.get(id); if(!task){toast("任务内容尚未加载，请刷新",true);return}
  if(task.task_type==="frontier"){
    switchView("frontier");$("#frontier-title").value=task.concept||task.title.replace("前沿待嫁接：","");$("#frontier-url").value=task.payload.url||"";$("#frontier-text").value=task.payload.summary||"";$("#analyze-form").scrollIntoView({behavior:"smooth"});return
  }
  activeReviewTask=task;reviewConversation=[];$("#review-title").textContent=task.title;const payload=task.payload||{};
  const question=payload.question||(payload.questions||[]).join("\n")||`请用自己的话解释“${task.concept}”的关键机制。`;$("#review-question").textContent=question;
  if(task.task_type==="quiz"){$("#review-answer-area").innerHTML=`<div class="quiz-options">${(payload.options||[]).map((option,index)=>`<label class="quiz-option"><input type="radio" name="quiz-answer" value="${index}"><span>${escapeHTML(option)}</span></label>`).join("")}</div>`;$("#rating-wrap").style.display="none"}else{$("#review-answer-area").innerHTML='<textarea id="review-text-answer" class="review-text" placeholder="先遮住答案，用自己的话写出理解；举例或说明边界会更有效。"></textarea>';$("#rating-wrap").style.display="block"}
  $("#review-result").className="review-result";$("#review-result").innerHTML="";$("#review-hint-box")?.remove();$("#review-form").style.display="block";$("#review-form button[type=submit]").textContent="提交回答";$("#review-modal").classList.add("show");$("#review-modal").setAttribute("aria-hidden","false");
}
window.openReview=openReview;
function closeReview(){$("#review-modal").classList.remove("show");$("#review-modal").setAttribute("aria-hidden","true");activeReviewTask=null}
$("#review-close").onclick=closeReview;$("#review-modal").onclick=e=>{if(e.target===$("#review-modal"))closeReview()};
$("#review-hint").onclick=async()=>{if(!activeReviewTask)return;try{const data=await api("/api/agent/hint",{method:"POST",body:JSON.stringify({task_id:activeReviewTask.id})});let box=$("#review-hint-box");if(!box){box=document.createElement("div");box.id="review-hint-box";box.className="hint-box";$("#review-hint").after(box)}box.textContent=data.result.hint}catch(err){toast(err.message,true)}};
$("#review-form").onsubmit=async e=>{e.preventDefault();if(!activeReviewTask)return;let answer;const following=reviewConversation.length>0;if(activeReviewTask.task_type==="quiz"&&!following){const selected=document.querySelector('input[name="quiz-answer"]:checked');if(!selected){toast("请先选择一个答案",true);return}answer=Number(selected.value)}else{answer=$("#review-text-answer")?.value.trim();if(!answer){toast("请先说说你的理解",true);return}}busy(true,"知识园丁正在理解你的思路……");try{const history=reviewConversation.map(item=>({role:item.role,content:item.content}));const data=await api("/api/tasks/answer",{method:"POST",body:JSON.stringify({id:activeReviewTask.id,answer,self_rating:Number($("#review-rating").value),history})});const r=data.result;reviewConversation.push({role:"user",content:String(answer)},{role:"assistant",content:`${r.understood||""}\n${r.feedback||""}\n${r.followup||""}`});$("#review-result").classList.add("show");if(!r.completed){$("#review-result").innerHTML=`<b>我先这样理解你的回答</b><p>${escapeHTML(r.understood||"你已经开始形成自己的解释。")}</p><p>${escapeHTML(r.feedback)}</p><div class="review-followup"><b>接着聊一步</b><p>${escapeHTML(r.followup)}</p></div>`;$("#review-question").textContent=r.followup;$("#review-answer-area").innerHTML='<textarea id="review-text-answer" class="review-text" placeholder="可以澄清、补充，也可以不同意园丁的理解。"></textarea>';$("#rating-wrap").style.display="block";$("#review-form button[type=submit]").textContent="继续回答";$("#review-text-answer").focus()}else{$("#review-form").style.display="none";$("#review-result").innerHTML=`<b>园丁已经理解到你的掌握 · +${r.earned_xp} XP</b><p>${escapeHTML(r.understood||"")}</p><p>${escapeHTML(r.feedback)}</p><p>下一次复习：${r.next_interval_days} 天后。间隔依据这段累计回答安排。</p><button class="secondary" id="review-done">返回花园</button>`;setTimeout(()=>{$("#review-done").onclick=async()=>{closeReview();await bootstrap();if($("#view-garden").classList.contains("active"))loadGarden()}},0)}}catch(err){toast(err.message,true)}finally{busy(false)}};

async function loadFrontierNotes() {
  try { const data = await api("/api/notes?kind=frontier"); const notes = data.notes.slice(0,8); window.frontierNotes = notes;
    $("#frontier-notes").innerHTML = notes.length ? notes.map((n,i) => `<div class="source-item"><div><b>${escapeHTML(n.title)}</b><small>${escapeHTML(n.source)} · ${escapeHTML(excerpt(n.content,55))}</small></div><button onclick="useFrontier(${i})">生成卡片</button></div>`).join("") : '<div class="empty">刷新关注源后，新文章会出现在这里。</div>';
  } catch(e) { toast(e.message,true); }
}
window.useFrontier = (index) => { const n = window.frontierNotes[index]; $("#frontier-title").value=n.title; $("#frontier-url").value=n.source_url||""; $("#frontier-text").value=n.content; $("#analyze-form").scrollIntoView({behavior:"smooth"}); };

async function loadGarden() {
  try { const [mindmap, graph, tasks, cards] = await Promise.all([api("/api/mindmap"),api("/api/graph"),api("/api/tasks"),api("/api/cards")]); graphData=graph; drawMindmap(mindmap); renderTasks("#all-tasks",tasks.tasks); $("#all-cards").innerHTML = cards.cards.length ? cards.cards.map(c=>`<div class="mini-card"><b>${escapeHTML(c.concept)}</b><small>${escapeHTML(c.frontier_title)} · ${c.questions.length} 个思考题</small></div>`).join("") : '<div class="empty">还没有对照卡。</div>'; } catch(e){toast(e.message,true)}
}

function drawMindmap(data){
  const svg=$("#knowledge-graph"),ns="http://www.w3.org/2000/svg",width=1500,centerX=750,colors=["#6c73a8","#d17a43","#4d91a8","#b75f55","#6d9b62","#9b72a8","#c49a3c","#568d7b"];let serial=0;
  const root=data.tree||{title:"我的知识花园",children:[]};const groups=(root.children||[]).map((node,index)=>({node,side:index%2===0?-1:1,color:colors[index%colors.length]}));svg.style.minWidth="1200px";
  const leafCount=node=>!node.children?.length?1:node.children.reduce((sum,child)=>sum+leafCount(child),0);const leftLeaves=groups.filter(g=>g.side<0).reduce((s,g)=>s+leafCount(g.node),0);const rightLeaves=groups.filter(g=>g.side>0).reduce((s,g)=>s+leafCount(g.node),0);const height=Math.max(620,Math.max(leftLeaves,rightLeaves)*34+90);svg.style.height=`${height}px`;svg.setAttribute("viewBox",`0 0 ${width} ${height}`);svg.innerHTML="";
  const records=[],branches=[],firstById=new Map();
  function place(node,side,depth,yStart,color,parent){const leaves=leafCount(node),span=leaves*34;let childCursor=yStart,childRecords=[];for(const child of (node.children||[])){const placed=place(child,side,depth+1,childCursor,color,null);childRecords.push(placed.record);branches.push({from:null,to:placed.record,color,depth:depth+1});childCursor+=placed.span}const y=childRecords.length?childRecords.reduce((s,r)=>s+r.y,0)/childRecords.length:yStart+span/2;const x=centerX+side*(155+(depth-1)*185);const record={key:`m${serial++}`,node,x,y,color,side,depth};records.push(record);if(typeof node.id==="number"&&!firstById.has(node.id))firstById.set(node.id,record);for(const branch of branches){if(branch.from===null&&childRecords.includes(branch.to))branch.from=record}return{record,span}}
  let leftY=40,rightY=40;const topRecords=[];for(const group of groups){const start=group.side<0?leftY:rightY;const placed=place(group.node,group.side,1,start,group.color,null);topRecords.push({...placed.record,top:true});branches.push({from:{x:centerX,y:height/2},to:placed.record,color:group.color,depth:1});if(group.side<0)leftY+=placed.span+24;else rightY+=placed.span+24}
  // Move each side as a block so its visual center aligns with the root.
  for(const side of [-1,1]){const sideRecords=records.filter(r=>r.side===side);if(!sideRecords.length)continue;const min=Math.min(...sideRecords.map(r=>r.y)),max=Math.max(...sideRecords.map(r=>r.y));const shift=height/2-(min+max)/2;sideRecords.forEach(r=>r.y+=shift)}
  for(const branch of branches){if(branch.from?.key){const actual=records.find(r=>r.key===branch.from.key);if(actual)branch.from=actual}const path=document.createElementNS(ns,"path");const sx=branch.from.x,sy=branch.from.y,tx=branch.to.x,ty=branch.to.y;const bend=(sx+tx)/2;path.setAttribute("d",`M ${sx} ${sy} C ${bend} ${sy}, ${bend} ${ty}, ${tx} ${ty}`);path.setAttribute("fill","none");path.setAttribute("stroke",branch.color);path.setAttribute("stroke-opacity",branch.depth===1?".8":".48");path.setAttribute("stroke-width",branch.depth===1?"5":branch.depth===2?"3":"1.7");path.setAttribute("class","mind-edge");svg.append(path)}
  for(const link of (data.cross_links||[])){if(link.status==="proposed"&&!showProposedCrossLinks)continue;const a=firstById.get(link.source_id),b=firstById.get(link.target_id);if(!a||!b)continue;const path=document.createElementNS(ns,"path");path.setAttribute("d",`M ${a.x} ${a.y} Q ${centerX} ${Math.min(a.y,b.y)-40}, ${b.x} ${b.y}`);path.setAttribute("fill","none");path.setAttribute("stroke","#d68a69");path.setAttribute("stroke-width",link.status==="accepted"?"2":"1.2");if(link.status==="proposed")path.setAttribute("stroke-dasharray","5 6");path.setAttribute("stroke-opacity",link.status==="accepted"?".72":".42");svg.append(path)}
  const rootGroup=document.createElementNS(ns,"g");const rootRect=document.createElementNS(ns,"rect");rootRect.setAttribute("x",centerX-78);rootRect.setAttribute("y",height/2-22);rootRect.setAttribute("width",156);rootRect.setAttribute("height",44);rootRect.setAttribute("rx",12);rootRect.setAttribute("fill","#2e684d");const rootText=document.createElementNS(ns,"text");rootText.setAttribute("x",centerX);rootText.setAttribute("y",height/2+5);rootText.setAttribute("text-anchor","middle");rootText.setAttribute("fill","white");rootText.setAttribute("font-size","15");rootText.setAttribute("font-weight","700");rootText.textContent=root.title;rootGroup.append(rootRect,rootText);svg.append(rootGroup);
  for(const record of records){const node=record.node,label=node.title||"知识点",boxWidth=Math.max(72,Math.min(190,label.length*12+24)),boxHeight=record.depth===1?32:27;const g=document.createElementNS(ns,"g");g.setAttribute("class","mind-node");g.style.cursor="pointer";const rect=document.createElementNS(ns,"rect");rect.setAttribute("x",record.x-boxWidth/2);rect.setAttribute("y",record.y-boxHeight/2);rect.setAttribute("width",boxWidth);rect.setAttribute("height",boxHeight);rect.setAttribute("rx",record.depth===1?8:5);rect.setAttribute("fill",record.depth===1?record.color:"#fffdf6");rect.setAttribute("stroke",record.color);rect.setAttribute("stroke-width",record.depth===1?"0":"1.4");const text=document.createElementNS(ns,"text");text.setAttribute("x",record.x);text.setAttribute("y",record.y+4);text.setAttribute("text-anchor","middle");text.setAttribute("fill",record.depth===1?"white":"#334b3d");text.setAttribute("font-size",record.depth===1?"12":"10.5");text.setAttribute("font-weight",record.depth<=2?"650":"500");text.textContent=label.length>16?label.slice(0,15)+"…":label;const tip=document.createElementNS(ns,"title");tip.textContent=node.summary||label;g.append(rect,text,tip);g.onclick=()=>showMindNode(node);svg.append(g)}
}

async function showMindNode(node){
  const host=$("#node-detail"),tags=(node.tags||[]).map(escapeHTML).join(" / ");host.innerHTML='<div class="empty">园丁正在读取这个节点的知识页……</div>';
  const byId=new Map(graphData.nodes.map(n=>[n.id,n])),unique=new Map();
  for(const edge of graphData.edges.filter(e=>e.source_id===node.id||e.target_id===node.id)){const otherId=edge.source_id===node.id?edge.target_id:edge.source_id;if(!unique.has(otherId))unique.set(otherId,edge)}
  const related=[...unique.values()].slice(0,5);let note=null;
  if(typeof node.id==="number"){try{note=(await api(`/api/note?id=${node.id}`)).note}catch(e){note=null}}
  const relationText=e=>e.relation==="contains"?"层级包含":e.relation==="wikilink"?"Wiki 双向链接":e.explanation||"候选语义关联";
  const links=related.map(e=>{const other=e.source_id===node.id?byId.get(e.target_id):byId.get(e.source_id);return `<div class="link-audit"><b>关联：${escapeHTML(other?.title||e.target_title)}</b><p>${escapeHTML(relationText(e))}</p></div>`}).join("");
  const pending=[...taskCache.values()].find(t=>t.status==="pending"&&t.concept===node.title);
  const activation=typeof node.knowledge_value==="number"?`<div class="knowledge-activation">知识活跃度 ${Math.round(node.knowledge_value*100)}% · 只影响相关结果排序，不会自动删除</div>`:"";
  host.innerHTML=`<div class="node-title"><b>${escapeHTML(node.title)}</b><span>${tags||escapeHTML(node.kind||"知识点")}</span></div>${activation}<p class="mind-summary">${escapeHTML(node.summary||"这是知识树中的一个学习节点。")}</p><div class="node-actions">${pending?'<button class="primary" id="node-review">用它复习</button>':""}<button class="secondary" id="node-ask">向园丁追问</button></div>${note?`<div class="node-content">${renderMarkdown(note.content)}</div>`:""}${links}`;
  if(pending)$("#node-review").onclick=()=>openReview(pending.id);
  $("#node-ask").onclick=()=>{switchView("home");$("#agent-ask-input").value=`请解释“${node.title}”的机制、例子和适用边界。`;$("#agent-ask-input").focus()};
}

function drawGraph(data) {
  const svg=$("#knowledge-graph"), width=svg.clientWidth||900, height=510; svg.setAttribute("viewBox",`0 0 ${width} ${height}`); svg.innerHTML="";
  if(!data.nodes.length){svg.innerHTML='<text x="50%" y="50%" text-anchor="middle" fill="#8c998e" font-size="12">同步 Obsidian 或导入教材后，知识树会在这里生长</text>';return}
  const nodes=data.nodes.map((n,i)=>({...n,x:width/2+Math.cos(i*2.4)*Math.min(width,height)*.32*Math.sqrt((i+3)/data.nodes.length),y:height/2+Math.sin(i*2.4)*height*.34*Math.sqrt((i+3)/data.nodes.length)})); const byId=new Map(nodes.map(n=>[n.id,n]));
  for(let step=0;step<65;step++){for(const n of nodes){let fx=(width/2-n.x)*.0015,fy=(height/2-n.y)*.0015;for(const m of nodes){if(n===m)continue;let dx=n.x-m.x,dy=n.y-m.y,d2=dx*dx+dy*dy+1;if(d2<7000){fx+=dx/d2*7;fy+=dy/d2*7}}n.x=Math.max(30,Math.min(width-30,n.x+fx));n.y=Math.max(30,Math.min(height-30,n.y+fy))}for(const e of data.edges){const a=byId.get(e.source_id),b=byId.get(e.target_id);if(!a||!b)continue;const dx=b.x-a.x,dy=b.y-a.y;a.x+=dx*.003;b.x-=dx*.003;a.y+=dy*.003;b.y-=dy*.003}}
  const ns="http://www.w3.org/2000/svg"; for(const e of data.edges){const a=byId.get(e.source_id),b=byId.get(e.target_id);if(!a||!b)continue;const line=document.createElementNS(ns,"line");line.setAttribute("x1",a.x);line.setAttribute("y1",a.y);line.setAttribute("x2",b.x);line.setAttribute("y2",b.y);line.setAttribute("class","graph-edge");line.setAttribute("stroke-width",Math.max(1,e.strength*2));if(e.cross_domain)line.setAttribute("stroke","#d48765");if(e.status==="proposed")line.setAttribute("stroke-dasharray","5 5");svg.append(line)}
  const colors={textbook:"#5c8b68",course:"#5c8b68",concept:"#5c8b68",domain:"#3d7655",moc:"#876e9d",source:"#a69a76",raw:"#a69a76",frontier:"#dda949",interest:"#d87e68",spark:"#d87e68",card:"#77a5a4",bridge:"#77a5a4"}; for(const n of nodes){const g=document.createElementNS(ns,"g");g.setAttribute("class","graph-node");const circle=document.createElementNS(ns,"circle");circle.setAttribute("cx",n.x);circle.setAttribute("cy",n.y);circle.setAttribute("r",n.kind==="domain"?11:["textbook","concept","course"].includes(n.kind)?7:9);circle.setAttribute("fill",colors[n.kind]||"#87a17d");circle.setAttribute("stroke","#fffdf5");circle.setAttribute("stroke-width","3");const label=document.createElementNS(ns,"text");label.setAttribute("x",n.x+11);label.setAttribute("y",n.y+3);label.setAttribute("class","graph-label");label.textContent=n.title.length>18?n.title.slice(0,17)+"…":n.title;g.append(circle,label);g.onclick=()=>showNode(n,data,byId);svg.append(g)}
}
function showNode(node,data,byId){const connected=data.edges.filter(e=>e.source_id===node.id||e.target_id===node.id);const linkRows=connected.slice(0,6).map(e=>{const other=e.source_id===node.id?byId.get(e.target_id):byId.get(e.source_id);const evidence=(e.evidence||[]).map(x=>`<small>依据：${escapeHTML(x)}</small>`).join("");const review=e.status==="proposed"?`<span class="review-actions"><button onclick="reviewLink(${e.id},true)">接受</button><button onclick="reviewLink(${e.id},false)">驳回</button></span>`:'<span class="verified">已确认</span>';return `<div class="link-audit"><b>→ ${escapeHTML(other?.title||e.target_title)}</b><em>${e.cross_domain?"跨学科":"同域"} · 置信度 ${Math.round((e.strength||0)*100)}%</em><p>${escapeHTML(e.explanation||"Obsidian 双向链接")}</p>${evidence}${e.relation==="semantic"?review:""}</div>`}).join("");$("#node-detail").innerHTML=`<div class="node-title"><b>${escapeHTML(node.title)}</b><span>${escapeHTML(node.kind)} · ${(node.tags||[]).map(escapeHTML).join(" / ")||"未分类"}</span></div>${linkRows||'<div class="empty">它还在等待一条新的连接。</div>'}`}
async function reviewLink(id,accepted){try{await api("/api/links/review",{method:"POST",body:JSON.stringify({id,accepted})});toast(accepted?"这条跨域根系已确认 +6 XP":"已驳回，园丁会尊重你的判断 +6 XP");await loadGarden();await bootstrap()}catch(e){toast(e.message,true)}}
window.reviewLink=reviewLink;

$$('.nav').forEach(n=>n.onclick=()=>switchView(n.dataset.view)); $$('[data-go]').forEach(n=>n.onclick=()=>switchView(n.dataset.go));
$("#show-cross-links").checked=showProposedCrossLinks;$("#show-cross-links").onchange=e=>{showProposedCrossLinks=e.target.checked;localStorage.setItem("garden.showProposedCrossLinks",showProposedCrossLinks?"1":"0");if($("#view-garden").classList.contains("active"))loadGarden()};
$("#refresh-all").onclick=async()=>{try{await bootstrap();toast("花园状态已刷新")}catch(e){toast(e.message,true)}};
$("#refresh-daily").onclick=()=>loadDaily(true);
function persistAgentConversation(){
  agentConversation = agentConversation.slice(-AGENT_HISTORY_LIMIT);
  localStorage.setItem("garden.agentConversation",JSON.stringify(agentConversation));
}
function renderAgentConversation(){
  const host=$("#agent-answer");
  if(!agentConversation.length){host.classList.remove("show");host.innerHTML="";return}
  host.classList.add("show");
  const hiddenCount=Math.max(0,agentConversation.length-AGENT_VISIBLE_MESSAGES);
  const visible=agentHistoryExpanded?agentConversation:agentConversation.slice(-AGENT_VISIBLE_MESSAGES);
  const toolbar=`<div class="agent-history-bar"><span>当前对话 ${agentConversation.length} 条 · 默认显示最近 ${Math.min(AGENT_VISIBLE_MESSAGES,agentConversation.length)} 条</span><div>${hiddenCount?`<button id="toggle-agent-history" class="text-btn" type="button">${agentHistoryExpanded?"收起较早记录":`查看更早 ${hiddenCount} 条`}</button>`:""}<button id="clear-agent-history" class="text-btn" type="button">清空当前对话</button></div></div>`;
  host.innerHTML=toolbar+visible.map((message,index)=>{
    if(message.role==="user")return `<div class="chat-turn user-turn"><small>你</small><p>${escapeHTML(withoutEmbeddedMaterial(message.content))}</p></div>`;
    const r=message.result||{};
    const latest=message===agentConversation.at(-1);
    const local=(r.citations||[]).map(c=>`《${escapeHTML(c.title)}》`).join("、")||"本轮沿用对话上下文";
    const online=(r.web_sources||[]).map((s,i)=>`<article class="research-source"><span>在线来源 ${i+1}${s.year?` · ${escapeHTML(s.year)}`:""}</span><a href="${escapeHTML(safeURL(s.url))}" target="_blank" rel="noopener noreferrer">${escapeHTML(s.title)}</a><small>${escapeHTML((s.authors||[]).slice(0,3).join("、")||s.venue||s.source||"")}</small></article>`).join("");
    const wechat=(r.wechat_sources||[]).map((s,i)=>`<article class="research-source"><span>授权微信依据 ${i+1}</span><b>${escapeHTML(s.title||s.talker||"本轮聊天片段")}</b><small>命中 ${escapeHTML(s.message_count||0)} 条 · ${escapeHTML(s.boundary||"")}</small></article>`).join("");
    const prompts=(r.discussion_prompts||[]).map(q=>`<button class="secondary discuss-prompt" type="button" data-prompt="${escapeHTML(q)}">${escapeHTML(q)}</button>`).join("");
    const saveAction=(r.wechat_sources||[]).length?'<button id="review-wechat-evidence" class="primary" type="button">打开微信苗圃审核原文</button>':'<button id="save-agent-insight" class="primary" type="button">沉淀并更新知识图谱</button>';
    const p=r.personalization||{};
    const statusLabels={applied:"已采用可信个性化",light:"仅轻量参考",standard:"标准讲解",disabled_first_exposure:"首次概览不个性化"};
    const personalizationEvidence=(p.evidence||[]).slice(0,4).map(item=>`<li><span>${escapeHTML(item.observation||"可追溯学习证据")}</span><small>${escapeHTML(item.evidence_id||"")} · 权重 ${Math.round(Number(item.weight||0)*100)}%</small></li>`).join("");
    const personalization=`<details class="personalization-audit"><summary>为什么这次这样讲 · ${escapeHTML(statusLabels[p.status]||"标准讲解")}${p.confidence?` ${Math.round(Number(p.confidence)*100)}%`:""}</summary><div><p>${escapeHTML(p.strategy_summary||p.fallback_reason||"本轮没有使用未经确认的画像判断。")}</p>${personalizationEvidence?`<ol>${personalizationEvidence}</ol>`:"<p class=\"muted\">没有足够相关证据，因此未启用个性化。</p>"}${r.request_id?`<div class="personalization-feedback"><span>这个讲法适合你吗？</span><button class="text-btn personalization-vote" data-request-id="${escapeHTML(r.request_id)}" data-helpful="true" type="button">适合</button><button class="text-btn personalization-vote" data-request-id="${escapeHTML(r.request_id)}" data-helpful="false" type="button">不适合</button></div>`:""}</div></details>`;
    const planner=renderPlannerAudit(r),visual=renderAgentVisualization(r.visualization||{});
    return `<div class="chat-turn assistant-turn"><small>园丁</small>${renderMarkdown(formatGardenerMarkdown(message.content))}${visual}<div class="sources">依据层：${escapeHTML(r.evidence_layer||"对话")} · 本地：${local}</div>${planner}${personalization}${online?`<div class="online-research"><b>为你找到的延伸阅读</b>${online}</div>`:""}${wechat?`<div class="online-research"><b>本轮授权读取的微信依据</b>${wechat}</div>`:""}${latest?`<div class="questions"><b>继续想一步</b><p>${escapeHTML(r.followup||"")}</p>${prompts}</div><div class="agent-decisions"><span>这段对话要怎样继续生长？</span>${saveAction}<button id="continue-agent-talk" class="secondary" type="button">继续讨论</button><button id="new-agent-talk" class="text-btn" type="button">开始新话题</button></div>`:""}</div>`;
  }).join("");
  $("#toggle-agent-history")?.addEventListener("click",()=>{agentHistoryExpanded=!agentHistoryExpanded;renderAgentConversation()});
  $("#clear-agent-history")?.addEventListener("click",startNewAgentConversation);
  $("#save-agent-insight")?.addEventListener("click",saveAgentInsight);
  $("#review-wechat-evidence")?.addEventListener("click",()=>{$('[data-view="wechat"]')?.click()});
  $("#continue-agent-talk")?.addEventListener("click",()=>continueAgentDiscussion(lastAgentExchange?.followup));
  $("#new-agent-talk")?.addEventListener("click",startNewAgentConversation);
  $$('.discuss-prompt').forEach(button=>button.onclick=()=>continueAgentDiscussion(button.dataset.prompt));
  $$('.personalization-vote').forEach(button=>button.onclick=()=>submitPersonalizationFeedback(button,button.dataset.helpful==="true"));
}
async function submitPersonalizationFeedback(button,helpful){
  const requestId=button.dataset.requestId;if(!requestId)return;
  button.closest('.personalization-feedback')?.querySelectorAll('button').forEach(item=>item.disabled=true);
  try{
    const data=await api('/api/agent/personalization-feedback',{method:'POST',body:JSON.stringify({request_id:requestId,helpful})});
    toast(data.result.message||'已记录你的反馈');
    if(!helpful){const input=$("#agent-ask-input");input.value="这次讲解方式不适合我。我希望你改成：";input.focus()}
  }catch(err){button.closest('.personalization-feedback')?.querySelectorAll('button').forEach(item=>item.disabled=false);toast(err.message,true)}
}
$("#agent-ask-form").onsubmit=async e=>{e.preventDefault();const input=$("#agent-ask-input"),question=input.value.trim();if(!question)return;const history=agentConversation.map(item=>({role:item.role,content:item.content})).slice(-10);agentConversation.push({role:"user",content:question});input.value="";persistAgentConversation();renderAgentConversation();busy(true,"知识园丁正在理解这段对话，必要时检索新证据……");try{const data=await api("/api/agent/ask",{method:"POST",body:JSON.stringify({question,history,session_id:agentSessionId})});const r=data.result;agentSessionId=r.session_id||agentSessionId;localStorage.setItem("garden.agentSessionId",agentSessionId);lastAgentExchange={question,...r};agentConversation.push({role:"assistant",content:r.answer,result:r});persistAgentConversation();renderAgentConversation()}catch(err){agentConversation.pop();persistAgentConversation();renderAgentConversation();toast(err.message,true)}finally{busy(false)}};
async function saveAgentInsight(){if(!lastAgentExchange||!agentConversation.length)return;busy(true,"正在把整段推导编译成知识点并重新嫁接思维导图……");try{const assistants=agentConversation.filter(item=>item.role==="assistant"&&item.result);const citations=assistants.flatMap(item=>item.result.citations||[]).filter((item,index,all)=>all.findIndex(other=>other.id===item.id)===index);const webSources=assistants.flatMap(item=>item.result.web_sources||[]).filter((item,index,all)=>all.findIndex(other=>other.url===item.url)===index);const firstQuestion=agentConversation.find(item=>item.role==="user")?.content||lastAgentExchange.question;const data=await api("/api/agent/save",{method:"POST",body:JSON.stringify({question:firstQuestion,answer:lastAgentExchange.answer,citations,web_sources:webSources,followup:lastAgentExchange.followup,messages:agentConversation.map(item=>({role:item.role,content:item.content}))})});toast(`已沉淀“${data.result.concept_title}”，知识库和思维导图已更新 +8 XP`);$("#save-agent-insight").disabled=true;$("#save-agent-insight").textContent="已沉淀";await bootstrap();await loadGarden()}catch(err){toast(err.message,true)}finally{busy(false)}}
function continueAgentDiscussion(prompt){const input=$("#agent-ask-input");input.value=prompt||"";input.focus();toast("参考方向已放入输入框，你可以修改或完全换成自己的问题")}
function startNewAgentConversation(){agentConversation=[];lastAgentExchange=null;agentHistoryExpanded=false;agentSessionId=crypto.randomUUID();localStorage.setItem("garden.agentSessionId",agentSessionId);persistAgentConversation();renderAgentConversation();$("#agent-ask-input").focus()}
$("#agent-patrol").onclick=async()=>{busy(true,"知识园丁正在巡视 raw、Wiki 和 AGENTS.md……");try{const data=await api("/api/agent/patrol",{method:"POST",body:"{}"});toast(`巡视完成：主动编译 ${data.result.ingested.length} 份资料，AGENTS.md 已同步`);await bootstrap();if($("#view-garden").classList.contains("active"))await loadGarden()}catch(err){toast(err.message,true)}finally{busy(false)}};
$("#analyze-form").onsubmit=async(e)=>{e.preventDefault();busy(true,"正在把前沿新枝嫁接到教材……");try{const data=await api("/api/analyze",{method:"POST",body:JSON.stringify({title:$("#frontier-title").value,url:$("#frontier-url").value,text:$("#frontier-text").value})});const cards=data.result.cards;$("#analysis-result").innerHTML='<h2 class="result-heading">新绽放的对照卡</h2>'+cards.map(c=>`<article class="bridge-card"><h3>✿ ${escapeHTML(c.concept)}</h3><div class="markdown">${renderMarkdown(c.explanation)}</div><div class="questions"><b>园丁追问</b><ol>${c.questions.map(q=>`<li>${escapeHTML(q)}</li>`).join("")}</ol></div></article>`).join("");toast(`已生成 ${cards.length} 张对照卡，并已嫁接到知识树`);await bootstrap();if($("#view-garden").classList.contains("active"))await loadGarden()}catch(err){toast(err.message,true)}finally{busy(false)}};
function inspirationText(result){const labels={fact:"事实",inference:"推测",imagination:"灵感",uncertain:"待核验"};return [result.acknowledgement,...(result.claims||[]).map(c=>`[${labels[c.status]||"待核验"}] ${c.text}`),result.counter_view?`[挑战视角] ${result.counter_view}`:""].filter(Boolean).join("\n\n")}
function persistInspiration(){localStorage.setItem("garden.inspirationConversation",JSON.stringify(inspirationConversation.slice(-20)));localStorage.setItem("garden.inspirationSessionId",inspirationSessionId)}
function renderInspiration(){const host=$("#capture-result");if(!inspirationConversation.length){host.innerHTML="";return}const turns=inspirationConversation.map(m=>m.role==="user"?`<div class="chat-turn user-turn"><small>你</small><p>${escapeHTML(m.content)}</p></div>`:`<div class="chat-turn assistant-turn"><small>灵感园丁 · ${escapeHTML(m.result?.primary_type||"开放探索")}</small>${renderMarkdown(m.content)}${(m.result?.anchors||[]).length?`<div class="sources">事实锚点：${m.result.anchors.map(a=>`《${escapeHTML(a.title)}》`).join("、")}</div>`:"<div class=\"sources\">本轮没有取得事实锚点；现实断言均标为待核验</div>"}</div>`).join("");const branches=(latestInspiration?.branches||[]).map(b=>`<button class="secondary inspiration-branch" type="button" data-question="${escapeHTML(b.question)}">${escapeHTML(b.title)}：${escapeHTML(b.question)}</button>`).join("");host.innerHTML=`<article class="panel inspiration-dialogue"><div class="mode-banner">💡 灵感模式 · 推测与想象不会进入掌握度或长期画像</div>${turns}<div class="inspiration-routes"><b>参考路标（可忽略，也可以完全自己输入）</b>${branches}</div><form id="inspiration-followup-form"><textarea id="inspiration-followup" rows="4" placeholder="继续写你自己的想法、质疑或新方向……"></textarea><button class="primary" type="submit">继续自由讨论</button></form><div class="agent-decisions"><button id="save-inspiration" class="secondary" type="button">保存未核验灵感种子</button><button id="investigate-inspiration" class="primary" type="button">建立中立领域概览</button><button id="verify-inspiration" class="secondary" type="button">只核验当前假设</button><button id="new-inspiration" class="text-btn" type="button">开始新灵感</button></div></article>`;$$('.inspiration-branch').forEach(button=>button.onclick=()=>{const input=$("#inspiration-followup");input.value=button.dataset.question;input.focus();toast("路标已填入，你可以任意修改后再发送")});$("#inspiration-followup-form").onsubmit=e=>{e.preventDefault();sendInspiration($("#inspiration-followup").value)};$("#save-inspiration").onclick=saveInspiration;$("#investigate-inspiration").onclick=()=>investigateInspiration("overview");$("#verify-inspiration").onclick=()=>investigateInspiration("verify");$("#new-inspiration").onclick=()=>{inspirationConversation=[];latestInspiration=null;inspirationSessionId=crypto.randomUUID();persistInspiration();renderInspiration();$("#capture-content").focus()}}
async function sendInspiration(message){message=String(message||"").trim();if(!message){toast("先写下你想继续探索的内容",true);return}const history=inspirationConversation.map(item=>({role:item.role,content:item.content}));inspirationConversation.push({role:"user",content:message});persistInspiration();renderInspiration();busy(true,"灵感园丁正在区分事实锚点、推测与想象……");try{const data=await api("/api/inspiration/ask",{method:"POST",body:JSON.stringify({message,history,session_id:inspirationSessionId})});latestInspiration=data.result;inspirationSessionId=data.result.session_id;inspirationConversation.push({role:"assistant",content:inspirationText(data.result),result:data.result});persistInspiration();renderInspiration()}catch(err){inspirationConversation.pop();persistInspiration();renderInspiration();toast(err.message,true)}finally{busy(false)}}
async function saveInspiration(){if(!latestInspiration)return;const tags=$("#capture-tags").value.split(/[，,、]/).map(x=>x.trim()).filter(Boolean);busy(true,"正在保存为未核验灵感种子……");try{const data=await api("/api/inspiration/save",{method:"POST",body:JSON.stringify({title:$("#capture-title").value,messages:inspirationConversation.map(x=>({role:x.role,content:x.content})),latest:latestInspiration,tags})});toast(`已保存“${data.result.title}”；它不会作为事实依据或掌握度证据 +5 XP`);await bootstrap()}catch(err){toast(err.message,true)}finally{busy(false)}}
function investigateInspiration(mode="overview"){
  if(!latestInspiration)return;
  const userTurns=inspirationConversation.filter(x=>x.role==="user").map(x=>x.content).filter(Boolean);
  const original=userTurns[0]||"";
  const latest=userTurns.at(-1)||original;
  const assumptions=(latestInspiration.assumptions||[]).join("；")||latest;
  const context=userTurns.slice(-5).map((text,index)=>`${index+1}. ${text}`).join("\n");
  switchView("home");
  const input=$("#agent-ask-input");
  const request=mode==="overview"
    ? "【领域概览】这是我第一次系统接触这个方向。请先给我一张准确、完整、中立、可追溯的领域地图，不关联我的专业、兴趣或旧笔记，也不替我选择路线；个性化留到我主动追问之后。"
    : "【严谨探究】我只想核验当前假设。请先准确复述命题，再判断需要核验哪些环节；只引用能直接支撑相应推理步骤的材料，不要为了连接知识库而硬套课本。";
  input.value=`${request}\n\n原始灵感：${original}\n当前追问：${latest}\n待核验假设：${assumptions}\n灵感对话中的用户思路：\n${context}`;
  input.focus();
  toast(mode==="overview"?"已准备中立领域概览；你仍可自由修改后再发送":"已带着当前假设转入严谨核验；你仍可自由修改后再发送")
}
$("#capture-form").onsubmit=async(e)=>{e.preventDefault();const message=$("#capture-content").value.trim();if(!message)return;await sendInspiration(message);$("#capture-content").value=""};
$("#settings-form").onsubmit=async(e)=>{e.preventDefault();busy(true,"正在同步 Obsidian 花圃……");try{const interests=$("#interests").value.split(/[，,、]/).map(x=>x.trim()).filter(Boolean);await api("/api/settings",{method:"POST",body:JSON.stringify({vault_path:$("#vault-path").value,learning_level:$("#learning-level").value,interests,textbook_directory:$("#textbook-directory").value})});if($("#vault-path").value){const result=await api("/api/sync",{method:"POST",body:JSON.stringify({vault_path:$("#vault-path").value})});toast(`同步完成：更新 ${result.result.changed} 篇笔记`)}else toast("学习画像与教材目录已保存");await bootstrap()}catch(err){toast(err.message,true)}finally{busy(false)}};
$("#wechat-config-form").onsubmit=async e=>{e.preventDefault();busy(true,"正在连接 TraceMemo，并验证 Token 与数据库状态……");try{const data=await api("/api/wechat/config",{method:"POST",body:JSON.stringify({base_url:$("#tracememo-base-url").value,token:$("#tracememo-token").value,save_token:$("#tracememo-save-token").checked})});$("#tracememo-token").value="";renderWechatStatus(data.status);toast(data.status.message,!data.status.authorized);await bootstrap()}catch(err){toast(err.message,true)}finally{busy(false)}};
$("#tracememo-forget").onclick=async()=>{try{const data=await api("/api/wechat/forget",{method:"POST",body:"{}"});renderWechatStatus(data.status);toast("已忘记本机保存的 TraceMemo Token");await bootstrap()}catch(err){toast(err.message,true)}};
$("#wechat-load-official-accounts").onclick=loadOfficialAccounts;
$("#wechat-official-form").onsubmit=async e=>{e.preventDefault();const talker=$("#wechat-official-account").value;if(!talker){toast("请先选择公众号",true);return}busy(true,"正在读取公众号文章卡片，并过滤普通聊天消息……");try{const data=await api("/api/wechat/articles",{method:"POST",body:JSON.stringify({talker,days:Number($("#wechat-official-days").value||30)})});renderOfficialArticles(data.result);$("#wechat-article-inbox").scrollIntoView({behavior:"smooth"})}catch(err){toast(err.message,true)}finally{busy(false)}};
$("#wechat-query-form").onsubmit=async e=>{e.preventDefault();busy(true,"正在按 Reader Skill 顺序确认本机时间、解析会话并读取消息……");try{const data=await api("/api/wechat/preview",{method:"POST",body:JSON.stringify({talker:$("#wechat-talker").value,time:$("#wechat-time").value})});renderWechatPreview(data.preview);$("#wechat-preview").scrollIntoView({behavior:"smooth"})}catch(err){toast(err.message,true)}finally{busy(false)}};
$("#wechat-recent").onclick=async()=>{try{const data=await api("/api/wechat/recent?limit=20"),items=candidateArray(data.result);$("#wechat-recent-result").innerHTML=items.length?`<div class="recent-chats">${items.map(item=>`<button class="secondary recent-chat" data-name="${escapeHTML(recentLabel(item))}">${escapeHTML(recentLabel(item))}</button>`).join("")}</div>`:'<div class="empty">TraceMemo 没有返回最近会话。</div>';$$('.recent-chat').forEach(button=>button.onclick=()=>{$("#wechat-talker").value=button.dataset.name;$("#wechat-talker").focus()})}catch(err){toast(err.message,true)}};
$("#wechat-refresh-candidates").onclick=loadWechatCandidates;
function closeReadingWorkspace(){
  $("#reading-workspace").classList.remove("show");
  $("#reading-workspace").setAttribute("aria-hidden","true");
}
$("#reading-close").onclick=closeReadingWorkspace;
$("#reading-question-form").onsubmit=async event=>{
  event.preventDefault();
  const input=$("#reading-question"), question=input.value.trim();
  if(!question)return;
  input.value="";
  await sendReadingQuestion(question,true);
};
document.addEventListener("keydown",event=>{if(event.key==="Escape"&&$("#reading-workspace").classList.contains("show"))closeReadingWorkspace()});
$("#import-textbooks").onclick=async()=>{const directory=$("#textbook-directory").value.trim();if(!directory){toast("请先填写教材总目录；目录中可以一次放入任意多本 PDF",true);return}busy(true,"正在增量扫描教材目录：未变化的教材会直接跳过……");try{const data=await api("/api/textbooks/import",{method:"POST",body:JSON.stringify({directory})});const r=data.result;const detail=r.failed?`，${r.failed} 本失败（${(r.errors||[])[0]?.error||"请检查 PDF"}）`:"";toast(`扫描完成：发现 ${r.files} 本，新增/变更 ${r.processed} 本，跳过 ${r.skipped} 本，提取 ${r.pages} 页${detail}`,Boolean(r.failed));await bootstrap()}catch(e){const message=e instanceof TypeError?"无法连接本地园丁服务。导入可能被中断；请确认启动窗口仍在运行后重试，已成功建立的索引不会丢失。":e.message;toast(`教材扫描失败：${message}`,true)}finally{busy(false)}};
$("#ingest-raw").onclick=async()=>{const raw_file=$("#raw-file").value.trim();if(!raw_file){toast("请填写 raw 文件的相对路径",true);return}busy(true,"正在把原始资料编译成互联 Wiki……");try{const data=await api("/api/ingest",{method:"POST",body:JSON.stringify({raw_file})});const r=data.result;$("#link-health").textContent=r.links.unresolved_count?`已编译，仍有 ${r.links.unresolved_count} 个链接待补全`:`已编译 ${r.concepts.length} 个概念，所有链接均已闭合`;toast(r.created.length?`新增 ${r.created.length} 个 Wiki 资产`:`该资料已编译，无需重复生成`);await api("/api/sync",{method:"POST",body:"{}"});await bootstrap()}catch(e){toast(e.message,true)}finally{busy(false)}};
$("#rebuild-links").onclick=async()=>{busy(true,"分类 Agent 正在理解待分类正文，再复核层级与跨域根系……");try{const data=await api("/api/links/rebuild",{method:"POST",body:"{}"});const h=data.hierarchy||{},c=data.classification||{};toast(`新归类 ${c.classified||0} 条，${c.needs_review||0} 条保留待复核；整理 ${h.relations||0} 条层级和 ${data.semantic||0} 条跨域连接`);await loadGarden();await bootstrap()}catch(e){toast(e.message,true)}finally{busy(false)}};
$("#feed-form").onsubmit=async(e)=>{e.preventDefault();try{const data=await api("/api/feeds",{method:"POST",body:JSON.stringify({name:$("#feed-name").value,url:$("#feed-url").value})});state.feeds=data.feeds;renderFeeds();e.target.reset();toast("关注源已添加")}catch(err){toast(err.message,true)}};
$("#refresh-feeds").onclick=async()=>{busy(true,"正在巡视关注的博主……");try{const data=await api("/api/feeds/refresh",{method:"POST",body:"{}"});toast(`发现 ${data.result.added} 篇新内容${data.result.errors.length?`，${data.result.errors.length} 个源失败`:""}`);await bootstrap();await loadFrontierNotes()}catch(e){toast(e.message,true)}finally{busy(false)}};
$("#copy-report").onclick=async()=>{const r=state.report;const text=`${r.title}\n\n${r.insight}\n\n新概念：${(r.new_concepts||[]).join("、")}\n下一步：${(r.next_actions||[]).join("；")}`;try{await navigator.clipboard.writeText(text);toast("周报已复制")}catch(e){toast("浏览器没有允许复制",true)}};

const hour=new Date().getHours();$("#page-title").textContent=`${hour<11?"早上":hour<18?"下午":"晚上"}好，园丁`;$("#today-label").textContent=new Intl.DateTimeFormat("zh-CN",{month:"long",day:"numeric",weekday:"long"}).format(new Date());
const restoredAssistant=[...agentConversation].reverse().find(item=>item.role==="assistant"&&item.result);const restoredQuestion=agentConversation.find(item=>item.role==="user")?.content;if(restoredAssistant)lastAgentExchange={question:restoredQuestion,...restoredAssistant.result};renderAgentConversation();
const restoredInspiration=[...inspirationConversation].reverse().find(item=>item.role==="assistant"&&item.result);if(restoredInspiration)latestInspiration=restoredInspiration.result;renderInspiration();
bootstrap().catch(e=>toast(`启动失败：${e.message}`,true));

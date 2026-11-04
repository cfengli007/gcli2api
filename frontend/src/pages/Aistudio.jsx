import {
  AlertCircle,
  CheckCircle,
  KeyRound,
  Loader2,
  Plus,
  Power,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import api from "../api";

const STATUS_STYLE = {
  ok: "bg-green-500/10 text-green-400 border-green-500/30",
  error: "bg-red-500/10 text-red-400 border-red-500/30",
  untested: "bg-dark-700 text-gray-400 border-dark-600",
};

function fmtTime(value) {
  if (!value) return "-";
  try {
    // 后端存的是秒级 Unix 时间戳；小于 1e12 一律按秒处理，乘 1000 转毫秒
    const ms = typeof value === "number" && value < 1e12 ? value * 1000 : Number(value) || 0;
    return new Date(ms).toLocaleString("zh-CN", { hour12: false });
  } catch {
    return String(value);
  }
}

export default function Aistudio() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState(null);
  const [keyText, setKeyText] = useState("");
  const [label, setLabel] = useState("");

  const showMessage = (type, text) => setMessage({ type, text });

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get("/aistudio/keys");
      setData(res.data);
    } catch (err) {
      showMessage("error", err.response?.data?.detail || "加载 AI Studio Key 失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const addKeys = async () => {
    if (!keyText.trim()) {
      showMessage("error", "请先粘贴至少一个 API Key");
      return;
    }
    setBusy("add");
    try {
      const res = await api.post("/aistudio/keys", {
        keys: keyText,
        label: label.trim(),
        verify: true,
      });
      setData(res.data);
      const added = res.data?.added?.length || 0;
      const skipped = res.data?.skipped?.length || 0;
      const failed = (res.data?.tested || []).filter((t) => !t.ok).length;
      let text = "添加完成：新增 " + added + " 个，跳过 " + skipped + " 个";
      if (failed) text += "，验证失败 " + failed + " 个";
      showMessage("success", text);
      if (added) {
        setKeyText("");
        setLabel("");
      }
    } catch (err) {
      showMessage("error", err.response?.data?.detail || "添加失败");
    } finally {
      setBusy("");
    }
  };

  const toggle = async (item) => {
    setBusy(item.id);
    try {
      const res = await api.post("/aistudio/keys/" + item.id + "/toggle", {
        enabled: !item.enabled,
      });
      setData(res.data);
      showMessage("success", "已" + (item.enabled ? "停用 " : "启用 ") + item.masked);
    } catch (err) {
      showMessage("error", err.response?.data?.detail || "操作失败");
    } finally {
      setBusy("");
    }
  };

  const remove = async (item) => {
    if (!window.confirm("确定删除 " + item.masked + "？")) return;
    setBusy(item.id);
    try {
      const res = await api.delete("/aistudio/keys/" + item.id);
      setData(res.data);
      showMessage("success", "已删除 " + item.masked);
    } catch (err) {
      showMessage("error", err.response?.data?.detail || "删除失败");
    } finally {
      setBusy("");
    }
  };

  const testOne = async (item) => {
    setBusy(item.id);
    try {
      const res = await api.post("/aistudio/keys/" + item.id + "/test");
      setData(res.data);
      const t = res.data?.test;
      if (t?.ok) {
        showMessage("success", item.masked + " 验证通过（" + t.latency + "s）");
      } else {
        showMessage("error", item.masked + " 验证失败：" + (t?.error || "未知错误"));
      }
    } catch (err) {
      showMessage("error", err.response?.data?.detail || "测试失败");
    } finally {
      setBusy("");
    }
  };

  const testAll = async () => {
    setBusy("all");
    try {
      const res = await api.post("/aistudio/test-all");
      setData(res.data);
      const tested = res.data?.tested || [];
      const ok = tested.filter((t) => t.ok).length;
      showMessage("success", "全部测试完成：" + ok + "/" + tested.length + " 通过");
    } catch (err) {
      showMessage("error", err.response?.data?.detail || "测试失败");
    } finally {
      setBusy("");
    }
  };

  const stats = data?.stats || { total: 0, enabled: 0, disabled: 0, error: 0 };
  const items = data?.items || [];
  const configured = !!data?.configured;
  const baseModel = data?.model || "gemini-3.5-flash-lite";
  const modelLines = [
    "抗截断-" + baseModel + "    流式抗截断（注入 emit_answer 合成工具 + 续写）",
    "假流式-" + baseModel + "    非流式取完整结果后分块回放",
    baseModel + "    原生流式",
  ].join("\n");

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/10 border border-purple-700/30 rounded-xl p-5">
        <h3 className="text-lg font-semibold text-purple-400 mb-2 flex items-center gap-2">
          <KeyRound size={18} />
          Gemini AI Studio API Key
        </h3>
        <p className="text-sm text-gray-400">
          通过 Google AI Studio 的 API Key 直连 generativelanguage.googleapis.com。
          Key 只保存在服务端 creds/aistudio_keys.json，列表接口仅返回打码值。
        </p>
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <div className="bg-dark-800/60 rounded-lg p-2.5">
            <div className="text-gray-500">当前模型</div>
            <div className="text-gray-200 font-mono truncate">{data?.model || "-"}</div>
          </div>
          <div className="bg-dark-800/60 rounded-lg p-2.5">
            <div className="text-gray-500">上游地址</div>
            <div className="text-gray-200 font-mono truncate">{data?.base_url || "-"}</div>
          </div>
          <div className="bg-dark-800/60 rounded-lg p-2.5">
            <div className="text-gray-500">已启用 / 总数</div>
            <div className="text-gray-200 font-mono">
              {stats.enabled} / {stats.total}
            </div>
          </div>
          <div className="bg-dark-800/60 rounded-lg p-2.5">
            <div className="text-gray-500">服务状态</div>
            <div className={configured ? "text-green-400" : "text-amber-400"}>
              {configured ? "可用" : "未配置 Key（接口 503）"}
            </div>
          </div>
        </div>
      </div>

      {message && (
        <div
          className={
            message.type === "error"
              ? "px-4 py-3 rounded-lg flex items-start gap-2 text-sm bg-red-500/10 border border-red-500/30 text-red-400"
              : "px-4 py-3 rounded-lg flex items-start gap-2 text-sm bg-green-500/10 border border-green-500/30 text-green-400"
          }
        >
          {message.type === "error" ? <AlertCircle size={16} /> : <CheckCircle size={16} />}
          <span className="flex-1">{message.text}</span>
        </div>
      )}

      <div className="card space-y-3">
        <h4 className="text-sm font-medium text-gray-300">添加 API Key</h4>
        <textarea
          value={keyText}
          onChange={(e) => setKeyText(e.target.value)}
          placeholder="每行一个 Key，也支持逗号或分号分隔"
          rows={4}
          className="w-full bg-dark-900 border border-dark-600 rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-purple-500"
        />
        <div className="flex flex-col sm:flex-row gap-2">
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="备注（可选，例如 主账号）"
            className="flex-1 bg-dark-900 border border-dark-600 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-purple-500"
          />
          <button
            onClick={addKeys}
            disabled={busy === "add"}
            className="btn btn-primary text-sm flex items-center justify-center gap-1.5 disabled:opacity-50"
          >
            {busy === "add" ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
            {busy === "add" ? "添加中..." : "添加并验证"}
          </button>
        </div>
        <p className="text-xs text-gray-500">
          添加时会并发调用一次 /v1beta/models 验证 Key 有效性；重复 Key 自动跳过。
        </p>
      </div>

      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium text-gray-300">Key 列表（{items.length}）</h4>
          <div className="flex gap-2">
            <button
              onClick={load}
              disabled={loading}
              className="btn btn-secondary text-sm flex items-center gap-1.5"
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
              刷新
            </button>
            <button
              onClick={testAll}
              disabled={busy === "all" || items.length === 0}
              className="btn btn-secondary text-sm flex items-center gap-1.5 disabled:opacity-50"
            >
              {busy === "all" ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <RefreshCw size={14} />
              )}
              全部测试
            </button>
          </div>
        </div>

        {loading ? (
          <div className="text-center text-gray-500 py-8 text-sm">
            <Loader2 size={18} className="animate-spin mx-auto mb-2" />
            加载中...
          </div>
        ) : items.length === 0 ? (
          <div className="text-center text-gray-500 py-8 text-sm">
            还没有添加 Key。没有可用 Key 时所有接口返回 503。
          </div>
        ) : (
          <div className="space-y-2">
            {items.map((item) => (
              <div
                key={item.id}
                className="bg-dark-800 border border-dark-600 rounded-lg p-3 flex flex-col sm:flex-row sm:items-center gap-3"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-sm text-gray-200">{item.masked}</span>
                    {item.label && (
                      <span className="text-xs px-2 py-0.5 rounded bg-dark-700 text-gray-400">
                        {item.label}
                      </span>
                    )}
                    <span
                      className={
                        "text-xs px-2 py-0.5 rounded border " +
                        (STATUS_STYLE[item.status] || STATUS_STYLE.untested)
                      }
                    >
                      {item.status}
                    </span>
                    <span
                      className={
                        "text-xs px-2 py-0.5 rounded border " +
                        (item.enabled
                          ? "bg-green-500/10 text-green-400 border-green-500/30"
                          : "bg-dark-700 text-gray-500 border-dark-600")
                      }
                    >
                      {item.enabled ? "启用" : "停用"}
                    </span>
                  </div>
                  <div className="mt-1 text-xs text-gray-500 flex flex-wrap gap-x-4 gap-y-1">
                    <span>最近使用：{fmtTime(item.last_used)}</span>
                    <span>最近检测：{fmtTime(item.last_checked)}</span>
                    {item.last_error && (
                      <span className="text-red-400">错误：{item.last_error}</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <button
                    onClick={() => testOne(item)}
                    disabled={busy === item.id}
                    className="btn btn-secondary text-xs px-2.5 py-1.5 disabled:opacity-50"
                  >
                    {busy === item.id ? <Loader2 size={13} className="animate-spin" /> : "测试"}
                  </button>
                  <button
                    onClick={() => toggle(item)}
                    disabled={busy === item.id}
                    className="btn btn-secondary text-xs px-2.5 py-1.5 flex items-center gap-1 disabled:opacity-50"
                  >
                    <Power size={13} />
                    {item.enabled ? "停用" : "启用"}
                  </button>
                  <button
                    onClick={() => remove(item)}
                    disabled={busy === item.id}
                    className="text-gray-500 hover:text-red-400 disabled:opacity-50"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card space-y-2">
        <h4 className="text-sm font-medium text-gray-300">可用模型名</h4>
        <p className="text-xs text-gray-500">
          代理只暴露 AISTUDIO_MODEL 一个基础模型，并以功能前缀提供变体：
        </p>
        <pre className="text-xs bg-dark-900 rounded-lg p-3 font-mono text-gray-300 whitespace-pre-wrap">
          {modelLines}
        </pre>
        <p className="text-xs text-gray-500">
          端点与 Antigravity 相同：/v1beta/models/&lt;model&gt;:generateContent、
          :streamGenerateContent、/v1/chat/completions、/v1/messages；客户端密码为面板设置的 API 密码。
        </p>
      </div>
    </div>
  );
}

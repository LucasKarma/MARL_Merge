import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";

const COLORS = {
  IDM: "#94a3b8",
  MARL: "#3b82f6", 
  "MARL+PIDM": "#10b981",
};

const collisionData = [
  { cooperation: "Cooperative", IDM: 100, MARL: 0, "MARL+PIDM": 0 },
  { cooperation: "Non-cooperative", IDM: 100, MARL: 0, "MARL+PIDM": 0 },
  { cooperation: "Mixed", IDM: 100, MARL: 7, "MARL+PIDM": 9 },
];

const rampRatioData = [
  { cooperation: "Cooperative", IDM: 0, MARL: 25, "MARL+PIDM": 50 },
  { cooperation: "Non-cooperative", IDM: 0, MARL: 50, "MARL+PIDM": 25 },
  { cooperation: "Mixed", IDM: 0, MARL: 72, "MARL+PIDM": 41 },
];

const speedData = [
  { cooperation: "Cooperative", IDM: 19.64, MARL: 22.68, "MARL+PIDM": 22.13 },
  { cooperation: "Non-cooperative", IDM: 19.17, MARL: 22.79, "MARL+PIDM": 21.95 },
  { cooperation: "Mixed", IDM: 19.43, MARL: 22.55, "MARL+PIDM": 22.04 },
];

const CustomTooltip = ({ active, payload, label, suffix }) => {
  if (!active || !payload) return null;
  return (
    <div style={{
      background: "#1e293b",
      border: "1px solid #334155",
      borderRadius: "8px",
      padding: "12px 16px",
      boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
    }}>
      <p style={{ color: "#e2e8f0", fontWeight: 600, marginBottom: 8, fontSize: 13 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, fontSize: 12, margin: "2px 0" }}>
          {p.name}: <strong>{p.value}{suffix}</strong>
        </p>
      ))}
    </div>
  );
};

const methods = ["IDM", "MARL", "MARL+PIDM"];

export default function ExperimentMatrix() {
  const [activeMetric, setActiveMetric] = useState("collision");

  const metrics = {
    collision: { data: collisionData, title: "Collision Rate (%)", suffix: "%", color: "#ef4444" },
    rampRatio: { data: rampRatioData, title: "Ramp Merge Ratio (%)", suffix: "%", color: "#10b981" },
    speed: { data: speedData, title: "Average Speed (m/s)", suffix: " m/s", color: "#3b82f6" },
  };

  const current = metrics[activeMetric];

  return (
    <div style={{
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      background: "#0f172a",
      minHeight: "100vh",
      padding: "32px 24px",
      color: "#e2e8f0",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 32 }}>
        <div style={{
          display: "inline-block",
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "#10b981",
          background: "rgba(16,185,129,0.1)",
          padding: "4px 12px",
          borderRadius: 4,
          marginBottom: 12,
        }}>P3 Experiment Results</div>
        <h1 style={{
          fontSize: 22,
          fontWeight: 700,
          color: "#f8fafc",
          margin: "8px 0 4px",
          letterSpacing: "-0.02em",
        }}>3×3 Evaluation Matrix</h1>
        <p style={{ fontSize: 13, color: "#94a3b8", margin: 0 }}>
          Hard difficulty · 100 episodes · Strict metrics (forbidden-based merge detection)
        </p>
      </div>

      {/* Metric Toggle */}
      <div style={{
        display: "flex",
        justifyContent: "center",
        gap: 8,
        marginBottom: 28,
      }}>
        {[
          { key: "collision", label: "Collision Rate" },
          { key: "rampRatio", label: "Ramp Merge Ratio" },
          { key: "speed", label: "Avg Speed" },
        ].map(m => (
          <button
            key={m.key}
            onClick={() => setActiveMetric(m.key)}
            style={{
              padding: "8px 18px",
              borderRadius: 6,
              border: activeMetric === m.key ? "1px solid #10b981" : "1px solid #334155",
              background: activeMetric === m.key ? "rgba(16,185,129,0.15)" : "transparent",
              color: activeMetric === m.key ? "#10b981" : "#94a3b8",
              fontSize: 13,
              fontWeight: 500,
              cursor: "pointer",
              transition: "all 0.2s",
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div style={{
        background: "#1e293b",
        borderRadius: 12,
        padding: "24px 16px 16px",
        border: "1px solid #334155",
        marginBottom: 24,
      }}>
        <h3 style={{
          fontSize: 14,
          fontWeight: 600,
          color: "#f8fafc",
          textAlign: "center",
          marginBottom: 20,
        }}>{current.title}</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={current.data} barCategoryGap="20%" barGap={4}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis
              dataKey="cooperation"
              tick={{ fill: "#94a3b8", fontSize: 12 }}
              axisLine={{ stroke: "#475569" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              domain={activeMetric === "speed" ? [15, 25] : [0, 100]}
            />
            <Tooltip content={<CustomTooltip suffix={current.suffix} />} />
            <Legend
              wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
              formatter={(value) => <span style={{ color: "#e2e8f0" }}>{value}</span>}
            />
            {methods.map(method => (
              <Bar
                key={method}
                dataKey={method}
                fill={COLORS[method]}
                radius={[4, 4, 0, 0]}
                maxBarSize={48}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Data Table */}
      <div style={{
        background: "#1e293b",
        borderRadius: 12,
        padding: "20px",
        border: "1px solid #334155",
        marginBottom: 24,
        overflowX: "auto",
      }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, color: "#f8fafc", marginBottom: 16 }}>
          Full Results Table
        </h3>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #475569" }}>
              <th style={{ padding: "8px 12px", textAlign: "left", color: "#94a3b8", fontWeight: 500 }}>Method</th>
              <th style={{ padding: "8px 12px", textAlign: "left", color: "#94a3b8", fontWeight: 500 }}>Cooperation</th>
              <th style={{ padding: "8px 12px", textAlign: "right", color: "#94a3b8", fontWeight: 500 }}>Collision</th>
              <th style={{ padding: "8px 12px", textAlign: "right", color: "#94a3b8", fontWeight: 500 }}>Ramp Ratio</th>
              <th style={{ padding: "8px 12px", textAlign: "right", color: "#94a3b8", fontWeight: 500 }}>Speed</th>
            </tr>
          </thead>
          <tbody>
            {["Cooperative", "Non-cooperative", "Mixed"].flatMap(coop =>
              methods.map((method, j) => {
                const ci = collisionData.findIndex(d => d.cooperation === coop);
                const collision = collisionData[ci][method];
                const ramp = rampRatioData[ci][method];
                const spd = speedData[ci][method];
                const isLast = j === 2;
                return (
                  <tr key={`${coop}-${method}`} style={{
                    borderBottom: isLast ? "1px solid #334155" : "none",
                  }}>
                    <td style={{
                      padding: "6px 12px",
                      color: COLORS[method],
                      fontWeight: 500,
                    }}>{method}</td>
                    <td style={{ padding: "6px 12px", color: "#cbd5e1" }}>{j === 0 ? coop : ""}</td>
                    <td style={{
                      padding: "6px 12px",
                      textAlign: "right",
                      color: collision === 0 ? "#10b981" : collision <= 10 ? "#fbbf24" : "#ef4444",
                      fontWeight: 600,
                    }}>{collision}%</td>
                    <td style={{
                      padding: "6px 12px",
                      textAlign: "right",
                      color: ramp >= 50 ? "#10b981" : ramp > 0 ? "#fbbf24" : "#94a3b8",
                      fontWeight: 600,
                    }}>{ramp}%</td>
                    <td style={{
                      padding: "6px 12px",
                      textAlign: "right",
                      color: "#cbd5e1",
                    }}>{spd.toFixed(2)}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Key Findings */}
      <div style={{
        background: "rgba(16,185,129,0.08)",
        borderRadius: 12,
        padding: "20px",
        border: "1px solid rgba(16,185,129,0.2)",
      }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, color: "#10b981", marginBottom: 12 }}>
          Key Findings
        </h3>
        <div style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.8 }}>
          <p style={{ margin: "0 0 8px" }}>
            <strong style={{ color: "#f8fafc" }}>IDM baseline:</strong> 100% collision across all conditions — learning-based control is essential.
          </p>
          <p style={{ margin: "0 0 8px" }}>
            <strong style={{ color: "#f8fafc" }}>MARL+P-IDM vs MARL:</strong> In cooperative traffic, P-IDM doubles ramp merge ratio (50% vs 25%) with 0% collision, demonstrating that interaction-aware HDV modeling enables more effective cooperative merging.
          </p>
          <p style={{ margin: "0 0 8px" }}>
            <strong style={{ color: "#f8fafc" }}>Safety-efficiency trade-off:</strong> In mixed traffic, both methods show low collision rates (7-9%), but training variance across runs remains high — a known limitation of DQN + local reward in multi-agent settings.
          </p>
        </div>
      </div>
    </div>
  );
}

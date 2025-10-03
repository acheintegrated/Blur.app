// src/components/Settings.tsx

import React, { useEffect, useState } from "react";
import { X } from "lucide-react";
import { RainbowGlow } from "./RainbowGlow";
import { useSettings } from "./SettingsContext";

interface SettingsProps {
  onClose: () => void;
}

export const Settings: React.FC<SettingsProps> = ({ onClose }) => {
  const { settings, updateSettings } = useSettings();
  // Filter the initial active tab to ensure it's one of the remaining tabs
  const [activeTab, setActiveTab] = useState<"user" | "fonts" | "about">("user");
  const [localSettings, setLocalSettings] = useState({ ...settings });
  const [saving, setSaving] = useState(false);

  const SAMPLE_TEXT = "Sphinx of black quartz, judge my vow. 0123456789 !@#$%";

  useEffect(() => {
    setLocalSettings({ ...settings });
  }, [settings]);

  const handleSave = async () => {
    try {
      setSaving(true);
      // Ensure we only save the fields we still care about.
      // TypeScript should enforce this based on the new Settings type, 
      // but we explicitly remove 'instructions' and 'memory' just in case the context still has them.
      const { instructions, memory, notifications, soundEffects, autoSave, saveInterval, ...restOfLocalSettings } = localSettings;
      await updateSettings(restOfLocalSettings);
      onClose();
    } finally {
      setSaving(false);
    }
  };

  const updateLocalSetting = <K extends keyof typeof localSettings>(
    key: K,
    value: (typeof localSettings)[K],
  ) => setLocalSettings((prev) => ({ ...prev, [key]: value }));

  const fontOptions = [
    { value: "Courier New", label: "Courier New" },
    { value: "ChanticleerRomanNF", label: "Chanticleer Roman NF" },
    { value: "PlayfairDisplay", label: "Playfair Display" },
    { value: "RedHatMono", label: "Red Hat Mono" },
    { value: "TheanoDidot", label: "Theano Didot" },
    { value: "TheanoOldStyle", label: "Theano Old Style" },
    { value: "Petrona", label: "Petrona" },
  ];
  
  // Define the tabs you want to keep
  const TABS_TO_KEEP: ("user" | "fonts" | "about")[] = ["user", "fonts", "about"];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 backdrop-blur-md bg-black bg-opacity-30" />

      <div className="relative bg-black/90 border border-zinc-900 w-full max-w-2xl shadow-2xl overflow-hidden">
        <div className="border-b border-zinc-900 p-4 flex justify-between items-center">
          <RainbowGlow className="text-white text-xl font-bold" dynamic={true}>
            🜃 settings
          </RainbowGlow>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors duration-150"
            aria-label="Close settings"
          >
            <X size={20} />
          </button>
        </div>

        <div className="flex h-[500px]">
          {/* Sidebar */}
          <div className="w-1/4 border-r border-zinc-900 p-4">
            <nav className="space-y-2">
              {/* Only map the tabs we want to keep */}
              {TABS_TO_KEEP.map((tab) => (
                <button
                  key={tab}
                  className={`w-full text-left px-3 py-2 rounded ${
                    activeTab === tab
                      ? "bg-zinc-900 text-white"
                      : "text-gray-300 hover:text-white hover:bg-zinc-900/60"
                  }`}
                  onClick={() => setActiveTab(tab)}
                >
                  {tab}
                </button>
              ))}
            </nav>
          </div>

          {/* Content */}
          <div className="w-3/4 p-6 overflow-y-auto">
            {activeTab === "user" && (
              <div className="space-y-6">
                <h2 className="text-white text-lg mb-4">about the meat-being</h2>

                <div className="space-y-2">
                  <label className="block text-gray-300">name</label>
                  <input
                    type="text"
                    value={localSettings.userName}
                    onChange={(e) => updateLocalSetting("userName", e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-purple-500"
                  />
                </div>

                {/* REMOVED: Notifications Toggle */}
                {/* REMOVED: Sound Effects Toggle */}
              </div>
            )}

            {/* REMOVED: {activeTab === "instructions" && (...) } */}
            {/* REMOVED: {activeTab === "memory" && (...) } */}

            {activeTab === "fonts" && (
              <div className="space-y-6">
                <h2 className="text-white text-lg mb-4">font settings</h2>

                <div className="space-y-2">
                  <label className="block text-gray-300">interface font</label>
                  <select
                    value={localSettings.interfaceFont}
                    onChange={(e) =>
                      updateLocalSetting("interfaceFont", e.target.value)
                    }
                    className="w-full bg-zinc-900 border border-zinc-900 px-3 py-2 text-white"
                    style={{ fontFamily: localSettings.interfaceFont }}
                  >
                    {fontOptions.map((f) => (
                      <option
                        key={f.value}
                        value={f.value}
                        style={{ fontFamily: f.value }}
                      >
                        {f.label}
                      </option>
                    ))}
                  </select>

                  <div className="mt-2 rounded border border-zinc-800 bg-zinc-900 p-4">
                    <div
                      style={{ fontFamily: localSettings.interfaceFont }}
                      className="text-xl text-white"
                    >
                      Interface Heading — Aa
                    </div>
                    <div
                      style={{ fontFamily: localSettings.interfaceFont }}
                      className="text-sm text-gray-300 mt-1"
                    >
                      {SAMPLE_TEXT}
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="block text-gray-300">body text font</label>
                  <select
                    value={localSettings.bodyFont}
                    onChange={(e) =>
                      updateLocalSetting("bodyFont", e.target.value)
                    }
                    className="w-full bg-zinc-900 border border-zinc-900 px-3 py-2 text-white"
                  >
                    {fontOptions.map((f) => (
                      <option
                        key={f.value}
                        value={f.value}
                        style={{ fontFamily: f.value }}
                      >
                        {f.label}
                      </option>
                    ))}
                  </select>

                  <div className="mt-2 rounded border border-zinc-800 bg-zinc-900 p-4">
                    <div
                      style={{ fontFamily: localSettings.bodyFont }}
                      className="text-xl text-white"
                    >
                      Body Heading — Aa
                    </div>
                    <p
                      style={{ fontFamily: localSettings.bodyFont }}
                      className="text-base text-gray-300 mt-1 leading-7"
                    >
                      {SAMPLE_TEXT}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "about" && (
              <div className="flex flex-col h-full w-full p-0">
                <div className="flex-grow flex items-center justify-center">
                  <div className="text-center">
                    <h1 className="text-6xl font-bold neon-dynamic-glow-text">
                      Blur.
                    </h1>
                    <p className="text-gray-500 mt-4 text-sm tracking-widest">
                      v0.0.0
                    </p>
                  </div>
                </div>
                <div className="pb-2 text-center">
                  <p className="text-gray-400 text-xs">
                    © 2023 acheintegrated ∴ blur. all rights reserved.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-zinc-900 p-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-zinc-900 text-gray-300 hover:bg-gray-700 transition-colors duration-150 mr-2"
            disabled={saving}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-purple-600 text-white hover:bg-teal-500 transition-colors duration-150 disabled:opacity-60"
            disabled={saving}
          >
            {saving ? "Saving…" : "Save Changes"}
          </button>
        </div>
      </div>
    </div>
  );
};
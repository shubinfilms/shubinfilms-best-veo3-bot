# Callback namespaces

All interactive buttons use the `namespace:action` format. The table below
lists the primary actions exposed in the UI.

| Namespace | Action(s) | Description |
|-----------|-----------|-------------|
| `menu`    | `profile`, `kb`, `photo`, `music`, `video`, `dialog`, `root` | Top-level navigation between modules. |
| `profile` | `transactions`, `promo`, `invite` | Secondary actions inside the profile card. |
| `kb`      | `open`, `templates`, `faq`, `lessons`, `examples` | Knowledge base navigation. |
| `banana`  | `add_photo`, `prompt`, `clear`, `templates`, `tpl:<slug>`, `start`, `restart`, `switch_engine`, `back` | Banana editor workflow. Templates use the `tpl:bg_remove`, `tpl:bg_studio`, `tpl:outfit_black`, `tpl:makeup_soft`, or `tpl:desk_clean` suffixes. |
| `music`   | `choose_mode`, `vocal`, `instrumental`, `start` | Suno music generation. |
| `dialog`  | `open`, `plain`, `pm`, `off` | AI dialog mode selection. |

Legacy callback payloads are mapped to the new format inside
`hub_router.LEGACY_ALIASES`, ensuring backward compatibility for older
clients.

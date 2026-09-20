// Syntax themes for Expressive Code, drawn from the identity tokens
// (docs/design/identity/tokens.md): code sits on `surface`, comments are `muted`, keywords take
// the one accent, everything else is body ink. Keep the hex values in step with starlight.css;
// scripts/check-contrast.mjs fails the build if they drift apart.

const theme = (name, type, { surface, text, muted, accent }) => ({
  name,
  type,
  colors: {
    'editor.background': surface,
    'editor.foreground': text,
  },
  tokenColors: [
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: muted, fontStyle: 'italic' } },
    {
      scope: ['keyword', 'storage', 'storage.type', 'storage.modifier', 'keyword.control', 'keyword.operator.new'],
      settings: { foreground: accent, fontStyle: 'bold' },
    },
    { scope: ['string', 'constant.numeric', 'constant.language', 'entity.name.function', 'variable'], settings: { foreground: text } },
  ],
});

export const monographLight = theme('monograph-light', 'light', {
  surface: '#f2eee4',
  text: '#1d1a16',
  muted: '#5c554b',
  accent: '#97196a',
});

export const monographDark = theme('monograph-dark', 'dark', {
  surface: '#221e17',
  text: '#ece6d8',
  muted: '#aaa190',
  accent: '#f29ad6',
});

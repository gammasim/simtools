// Select the version shown in the switcher from the current site URL.
const versionInUrl = window.location.pathname.match(/\/(v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?)\//);
DOCUMENTATION_OPTIONS.theme_switcher_version_match = versionInUrl ? versionInUrl[1] : "latest";

// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Bridging Digital AI with Physical Robotics',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://robo-ai-book.vercel.app', // Replace with your actual Vercel URL
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub Pages: https://<USERNAME>.github.io/<REPO>/
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub Pages, you don't need these.
  organizationName: 'GIAIC', // Usually your GitHub org/user name.
  projectName: 'Robo-AI-Book', // Usually your repo name.
  deploymentBranch: 'gh-pages', // Branch that GitHub Pages will deploy from.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/GIAIC/Robo-AI-Book/tree/master/',
        },
        blog: false, // Disable blog if not needed
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'RoboAI Book',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            type: 'localeDropdown',
            position: 'right',
          },
          {
            href: 'https://github.com/GIAIC/Robo-AI-Book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'Module 1: ROS 2 Basics',
                to: '/docs/module-1/week-1',
              },
              {
                label: 'Module 2: Gazebo & Unity',
                to: '/docs/module-2/week-6',
              },
              {
                label: 'Module 3: NVIDIA Isaac',
                to: '/docs/module-3/week-8',
              },
              {
                label: 'Module 4: VLA Integration',
                to: '/docs/module-4/week-11',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/GIAIC/Robo-AI-Book',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GIAIC',
                href: 'https://giaic.org',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Global AI Council (GIAIC). Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'json', 'yaml'],
      },
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: true,
        },
      },
    }),

  plugins: [
    // Add any additional plugins here
    // Plugin for RAG chatbot integration (commented out since api/ doesn't contain docs)
    // [
    //   '@docusaurus/plugin-content-docs',
    //   {
    //     id: 'api',
    //     path: 'api',
    //     routeBasePath: 'api',
    //   },
    // ],
  ],

  stylesheets: [
    {
      href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
      type: 'text/css',
      rel: 'stylesheet',
    },
  ],
};

export default config;
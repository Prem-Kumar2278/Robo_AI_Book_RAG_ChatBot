// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: '📚 Preface',
      items: [
        'preface/week-0'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 1: ROS 2 Basics',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Introduction to ROS 2',
          items: [
            'module-1/week-1/index',
            {
              type: 'category',
              label: 'Lessons',
              items: [
                'module-1/week-1/lessons/introduction-to-ros2',
                'module-1/week-1/lessons/ros2-architecture',
                'module-1/week-1/lessons/setting-up-workspace',
                'module-1/week-1/lessons/creating-first-package',
                'module-1/week-1/lessons/basic-ros2-concepts',
                'module-1/week-1/lessons/running-examples',
                'module-1/week-1/lessons/quality-of-service',
                'module-1/week-1/lessons/exercises',
                'module-1/week-1/lessons/code-example',
                'module-1/week-1/lessons/ethical-considerations',
                'module-1/week-1/lessons/summary',
                'module-1/week-1/lessons/references',
              ],
            },
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Nodes and Topics',
          items: [
            'module-1/week-2/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: Services and Actions',
          items: [
            'module-1/week-3/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 4-5: Advanced ROS 2 Concepts',
          items: [
            'module-1/week-4/index',
            'module-1/week-5/index',
          ],
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part 2: Gazebo & Unity',
      items: [
        {
          type: 'category',
          label: 'Chapter 6: Gazebo Simulation',
          items: [
            'module-2/week-6/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 7: Unity Integration',
          items: [
            'module-2/week-7/index',
          ],
        },
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Part 3: NVIDIA Isaac',
      items: [
        {
          type: 'category',
          label: 'Chapter 8: Isaac ROS Fundamentals',
          items: [
            'module-3/week-8/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 9: Perception Pipeline',
          items: [
            'module-3/week-9/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 10: Navigation Pipeline',
          items: [
            'module-3/week-10/index',
          ],
        },
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Part 4: VLA Integration',
      items: [
        {
          type: 'category',
          label: 'Chapter 11: Vision-Language-Action Models',
          items: [
            'module-4/week-11/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 12: Integration & Control',
          items: [
            'module-4/week-12/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 13: Capstone Project',
          items: [
            'module-4/week-13/index',
          ],
        },
      ],
      collapsed: true,
    },
  ],
};

module.exports = sidebars;
import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI & Humanoid Robotics',
    description: (
      <>
        Learn how to bridge digital AI with physical robotics through our comprehensive curriculum.
      </>
    ),
  },
  {
    title: 'Modern Technologies',
    description: (
      <>
        Master ROS 2, Gazebo, NVIDIA Isaac, Unity, and Vision-Language-Action models.
      </>
    ),
  },
  {
    title: 'Hands-On Learning',
    description: (
      <>
        Practical exercises and capstone projects to solidify your understanding.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <div className={styles.featureSvg}>
          <div style={{fontSize: '2rem', marginBottom: '1rem'}}>🤖</div>
        </div>
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
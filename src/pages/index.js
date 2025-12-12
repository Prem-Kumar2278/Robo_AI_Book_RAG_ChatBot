import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/preface/week-0">
            Get Start
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Bridging Digital AI with Physical Robotics">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container padding-vert--lg">
            <div className="row">
              <div className="col col--4 padding-horiz--md">
                <div className={styles.featureCard}>
                  <Heading as="h3">🤖 Physical AI</Heading>
                  <p>Learn how intelligence emerges from the interaction between an agent and its environment.</p>
                </div>
              </div>
              <div className="col col--4 padding-horiz--md">
                <div className={styles.featureCard}>
                  <Heading as="h3">🦾 Humanoid Robotics</Heading>
                  <p>Master the technologies behind advanced humanoid robots using ROS 2, NVIDIA Isaac, and more.</p>
                </div>
              </div>
              <div className="col col--4 padding-horiz--md">
                <div className={styles.featureCard}>
                  <Heading as="h3">🎓 Comprehensive Curriculum</Heading>
                  <p>Four modules spanning 13 weeks of hands-on learning with practical projects.</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
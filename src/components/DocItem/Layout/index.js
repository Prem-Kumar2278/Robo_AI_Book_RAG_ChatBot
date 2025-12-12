import React from 'react';
import clsx from 'clsx';
import {
  PageMetadata,
  HtmlClassNameProvider,
  ThemeClassNames,
  useDoc,
} from '@docusaurus/theme-common';
import {
  DocProvider,
  useDocRouteMetadata,
} from '@docusaurus/plugin-content-docs/client';
import DocItemPaginator from '@theme/DocItemPaginator';
import DocVersionBanner from '@theme/DocVersionBanner';
import DocVersionBadge from '@theme/DocVersionBadge';
import DocItemFooter from '@theme/DocItemFooter';
import DocItemContent from '@theme/DocItemContent';
import DocBreadcrumbs from '@theme/DocBreadcrumbs';
import styles from './styles.module.css';

export default function DocItem(props) {
  const {content: DocContent} = props;
  const {metadata, frontMatter} = useDoc();
  const {wrapperClassName} = useDocRouteMetadata();

  const {
    hide_title: hideTitle,
    hide_table_of_contents: hideTableOfContents,
    table_of_contents: tableOfContents,
    wrapperClassName: metaWrapperClassName,
  } = frontMatter;

  const {description, title} = metadata;

  return (
    <HtmlClassNameProvider className="doc">
      <PageMetadata
        title={title}
        description={description}
        keywords={metadata.keywords}
      />
      <div className={clsx(wrapperClassName, metaWrapperClassName)}>
        <div className="container margin-vert--lg">
          <div className="row">
            <div className={clsx('col', {'col--8': !hideTableOfContents})}>
              <div className="doc-markdown">
                <article>
                  <DocVersionBanner />
                  <DocBreadcrumbs />
                  <DocVersionBadge />
                  {!hideTitle && (
                    <header>
                      <h1 className={styles.docTitle}>{title}</h1>
                    </header>
                  )}
                  <div className="margin-vert--lg">
                    <DocItemContent>
                      <DocContent />
                    </DocItemContent>
                  </div>
                  <footer>
                    <DocItemFooter />
                  </footer>
                </article>
                <div className="margin-vert--lg">
                  <DocItemPaginator />
                </div>
              </div>
            </div>
            {!hideTableOfContents && tableOfContents && (
              <div className="col col--3 col--offset-1">
                <div className="table-of-contents table-of-contents--left">
                  {/* Render the TOC */}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </HtmlClassNameProvider>
  );
}
/*
 *  Player - One Hell of a Robot Server
 *  Copyright (C) 2000  Brian Gerkey   &  Kasper Stoy
 *                      gerkey@usc.edu    kaspers@robotics.usc.edu
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */
/**************************************************************************
 * Desc: KD tree functions
 * Author: Andrew Howard
 * Date: 18 Dec 2002
 * CVS: $Id: pf_kdtree.h 6532 2008-06-11 02:45:56Z gbiggs $
 *************************************************************************/

#ifndef PF_KDTREE_H
#define PF_KDTREE_H

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <state/state_rep.h>

// Info for a node in the tree
typedef struct pf_kdtree_node
{
  // Depth in the tree
  int leaf, depth;

  // Pivot dimension and value
  int pivot_dim;
  double pivot_value;

  // The key for this node
  int key[3];

  // The value for this node
  double value;

  // The cluster label (leaf nodes)
  int cluster;

  // Child nodes
  struct pf_kdtree_node *children[2];

} pf_kdtree_node_t;


// A kd tree
typedef struct
{
  // Cell size
  double size[3];

  // The root node of the tree
  pf_kdtree_node_t *root;

  // The number of nodes in the tree
  int node_count, node_max_count;
  pf_kdtree_node_t *nodes;

  // The number of leaf nodes in the tree
  int leaf_count;

} pf_kdtree_t;

// Compare keys to see if they are equal
static int pf_kdtree_equal(pf_kdtree_t *self, int key_a[], int key_b[]);

// Insert a node into the tree
static pf_kdtree_node_t *pf_kdtree_insert_node(pf_kdtree_t *self, pf_kdtree_node_t *parent,
                                               pf_kdtree_node_t *node, int key[], double value);

// Recursive node search
static pf_kdtree_node_t *pf_kdtree_find_node(pf_kdtree_t *self, pf_kdtree_node_t *node, int key[]);

// Recursively label nodes in this cluster
static void pf_kdtree_cluster_node(pf_kdtree_t *self, pf_kdtree_node_t *node, int depth);

// Recursive node printing
//static void pf_kdtree_print_node(pf_kdtree_t *self, pf_kdtree_node_t *node);


////////////////////////////////////////////////////////////////////////////////
// Create a tree
pf_kdtree_t *pf_kdtree_alloc(int max_size)
{
  pf_kdtree_t *self;

  self = (pf_kdtree_t *)calloc(1, sizeof(pf_kdtree_t));

  self->size[0] = 0.50;
  self->size[1] = 0.50;
  self->size[2] = (10 * M_PI / 180);

  self->root = NULL;

  self->node_count = 0;
  self->node_max_count = max_size;
  self->nodes = (pf_kdtree_node_t *)calloc(self->node_max_count, sizeof(pf_kdtree_node_t));

  self->leaf_count = 0;

  return self;
}


////////////////////////////////////////////////////////////////////////////////
// Destroy a tree
void pf_kdtree_free(pf_kdtree_t *self)
{
  free(self->nodes);
  free(self);
  return;
}


////////////////////////////////////////////////////////////////////////////////
// Clear all entries from the tree
void pf_kdtree_clear(pf_kdtree_t *self)
{
  self->root = NULL;
  self->leaf_count = 0;
  self->node_count = 0;

  return;
}


////////////////////////////////////////////////////////////////////////////////
// Insert a pose into the tree.
void pf_kdtree_insert(pf_kdtree_t *self, PoseState pose, double value)
{
    int key[3];

    key[0] = floor(pose.pose_.x_ / self->size[0]);
    key[1] = floor(pose.pose_.y_ / self->size[1]);
    key[2] = floor(pose.rot_.getRPY().z_ / self->size[2]); // yaw
    self->root = pf_kdtree_insert_node(self, NULL, self->root, key, value);

    return;
}


////////////////////////////////////////////////////////////////////////////////
// Determine the probability estimate for the given pose. TODO: this
// should do a kernel density estimate rather than a simple histogram.
double pf_kdtree_get_prob(pf_kdtree_t *self, PoseState pose)
{
    int key[3];
    pf_kdtree_node_t *node;

    key[0] = floor(pose.pose_.x_ / self->size[0]);
    key[1] = floor(pose.pose_.y_ / self->size[1]);
    key[2] = floor(pose.rot_.getRPY().z_ / self->size[2]); // yaw

    node = pf_kdtree_find_node(self, self->root, key);
    if (node == NULL)
        return 0.0;
    return node->value;
}


////////////////////////////////////////////////////////////////////////////////
// Determine the cluster label for the given pose
int pf_kdtree_get_cluster(pf_kdtree_t *self, PoseState pose)
{
    int key[3];
    pf_kdtree_node_t *node;

    key[0] = floor(pose.pose_.x_ / self->size[0]);
    key[1] = floor(pose.pose_.y_ / self->size[1]);
    key[2] = floor(pose.rot_.getRPY().z_ / self->size[2]); // yaw
    node = pf_kdtree_find_node(self, self->root, key);
    if (node == NULL)
        return -1;
    return node->cluster;
}


////////////////////////////////////////////////////////////////////////////////
// Compare keys to see if they are equal
int pf_kdtree_equal(pf_kdtree_t *self, int key_a[], int key_b[])
{
  //double a, b;

  if (key_a[0] != key_b[0])
    return 0;
  if (key_a[1] != key_b[1])
    return 0;

  if (key_a[2] != key_b[2])
    return 0;

  return 1;
}


////////////////////////////////////////////////////////////////////////////////
// Insert a node into the tree
pf_kdtree_node_t *pf_kdtree_insert_node(pf_kdtree_t *self, pf_kdtree_node_t *parent,
                                        pf_kdtree_node_t *node, int key[], double value)
{
    int i;
    int split, max_split;    

    // If the node doesnt exist yet...
    if (node == NULL)
    {
        assert(self->node_count < self->node_max_count);

        node = self->nodes + self->node_count++;

        memset(node, 0, sizeof(node));

        node->leaf = 1;

        if (parent == NULL)
            node->depth = 0;
        else
            node->depth = parent->depth + 1;
        for (i = 0; i < 3; i++)
            node->key[i] = key[i];

        node->value = value;
        self->leaf_count += 1;
    }

    // If the node exists, and it is a leaf node...
    else if (node->leaf)
    {
        // If the keys are equal, increment the value
        if (pf_kdtree_equal(self, key, node->key))
        {
        node->value += value;
        }

        // The keys are not equal, so split this node
        else
        {
            // Find the dimension with the largest variance and do a mean
            // split
            max_split = 0;
            node->pivot_dim = -1;
            for (i = 0; i < 3; i++)
            {
                split = abs(key[i] - node->key[i]);
                if (split > max_split)
                {
                max_split = split;
                node->pivot_dim = i;
                }
            }
            assert(node->pivot_dim >= 0);

            node->pivot_value = (key[node->pivot_dim] + node->key[node->pivot_dim]) / 2.0;
            if (key[node->pivot_dim] < node->pivot_value)
            {
                node->children[0] = pf_kdtree_insert_node(self, node, NULL, key, value);
                node->children[1] = pf_kdtree_insert_node(self, node, NULL, node->key, node->value);
            }
            else
            {
                node->children[0] = pf_kdtree_insert_node(self, node, NULL, node->key, node->value);
                node->children[1] = pf_kdtree_insert_node(self, node, NULL, key, value);
            }

            node->leaf = 0;
            self->leaf_count -= 1;
        }
    }

    // If the node exists, and it has children...
    else
    {
        assert(node->children[0] != NULL);
        assert(node->children[1] != NULL);

        if (key[node->pivot_dim] < node->pivot_value)
        pf_kdtree_insert_node(self, node, node->children[0], key, value);
        else
        pf_kdtree_insert_node(self, node, node->children[1], key, value);
    }
    return node;
}


////////////////////////////////////////////////////////////////////////////////
// Recursive node search
pf_kdtree_node_t *pf_kdtree_find_node(pf_kdtree_t *self, pf_kdtree_node_t *node, int key[])
{
    if (node->leaf)
    {
        // printf("find  : leaf %p %d %d %d\n", node, node->key[0], node->key[1], node->key[2]);

        // If the keys are the same...
        if (pf_kdtree_equal(self, key, node->key))
        return node;
        else
        return NULL;
    }
    else
    {
        // printf("find  : brch %p %d %f\n", node, node->pivot_dim, node->pivot_value);

        assert(node->children[0] != NULL);
        assert(node->children[1] != NULL);

        // If the keys are different...
        if (key[node->pivot_dim] < node->pivot_value)
        return pf_kdtree_find_node(self, node->children[0], key);
        else
        return pf_kdtree_find_node(self, node->children[1], key);
    }

    return NULL;
}


////////////////////////////////////////////////////////////////////////////////
// Recursive node printing
/*
void pf_kdtree_print_node(pf_kdtree_t *self, pf_kdtree_node_t *node)
{
  if (node->leaf)
  {
    printf("(%+02d %+02d %+02d)\n", node->key[0], node->key[1], node->key[2]);
    printf("%*s", node->depth * 11, "");
  }
  else
  {
    printf("(%+02d %+02d %+02d) ", node->key[0], node->key[1], node->key[2]);
    pf_kdtree_print_node(self, node->children[0]);
    pf_kdtree_print_node(self, node->children[1]);
  }
  return;
}
*/


////////////////////////////////////////////////////////////////////////////////
// Cluster the leaves in the tree
void pf_kdtree_cluster(pf_kdtree_t *self)
{
    int i;
    int queue_count, cluster_count;
    pf_kdtree_node_t **queue, *node;

    queue_count = 0;
    queue = (pf_kdtree_node_t **)calloc(self->node_count, sizeof(queue[0]));

    // Put all the leaves in a queue
    for (i = 0; i < self->node_count; i++)
    {
        node = self->nodes + i;
        if (node->leaf)
        {
        node->cluster = -1;
        assert(queue_count < self->node_count);
        queue[queue_count++] = node;

        // TESTING; remove
        assert(node == pf_kdtree_find_node(self, self->root, node->key));
        }
    }

    cluster_count = 0;

    // Do connected components for each node
    while (queue_count > 0)
    {
        node = queue[--queue_count];

        // If this node has already been labelled, skip it
        if (node->cluster >= 0)
        continue;

        // Assign a label to this cluster
        node->cluster = cluster_count++;

        // Recursively label nodes in this cluster
        pf_kdtree_cluster_node(self, node, 0);
    }

    free(queue);
    return;
}


////////////////////////////////////////////////////////////////////////////////
// Recursively label nodes in this cluster
void pf_kdtree_cluster_node(pf_kdtree_t *self, pf_kdtree_node_t *node, int depth)
{
    int i;
    int nkey[3];
    pf_kdtree_node_t *nnode;

    for (i = 0; i < 3 * 3 * 3; i++)
    {
        nkey[0] = node->key[0] + (i / 9) - 1;
        nkey[1] = node->key[1] + ((i % 9) / 3) - 1;
        nkey[2] = node->key[2] + ((i % 9) % 3) - 1;

        nnode = pf_kdtree_find_node(self, self->root, nkey);
        if (nnode == NULL)
        continue;

        assert(nnode->leaf);

        // This node already has a label; skip it.  The label should be
        // consistent, however.
        if (nnode->cluster >= 0)
        {
        assert(nnode->cluster == node->cluster);
        continue;
        }

        // Label this node and recurse
        nnode->cluster = node->cluster;

        pf_kdtree_cluster_node(self, nnode, depth + 1);
    }
    return;
}

#endif